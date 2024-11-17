import pandas as pd
import numpy as np
import ccxt
import time
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import talib as ta  # Teknik analiz için TA-Lib kütüphanesi
from config import API_KEY, API_SECRET, TELEGRAM_TOKEN, CHAT_ID  # Konfigürasyon dosyasından API anahtarları

# Yapımcı: Kemal Buyukburc

# Binance API bağlantısı
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})

# Veriyi Yükleme ve Ön İşleme
def load_and_preprocess_data(filename="bitcoin_data.csv"):
    data = pd.read_csv(filename)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Sadece kapanış fiyatını kullanıyoruz (price prediction için)
    close_data = data[['timestamp', 'close']].set_index('timestamp')
    
    # Veriyi normalize et
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_data_scaled = scaler.fit_transform(close_data[['close']])
    
    return close_data_scaled, scaler

# Veriyi Zaman Serisi Formatına Çevirme
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    X = np.array(X)
    y = np.array(y)
    
    # Şekil uyumu
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

# LSTM Modeli Kurma ve Eğitme
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return model

# RSI, EMA, MACD gibi göstergelerle sinyaller eklemek
def calculate_indicators(data):
    rsi = ta.RSI(data['close'], timeperiod=14)  # RSI
    ema_short = ta.EMA(data['close'], timeperiod=12)  # Kısa vadeli EMA
    ema_long = ta.EMA(data['close'], timeperiod=26)  # Uzun vadeli EMA
    macd, macdsignal, macdhist = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
    
    data['RSI'] = rsi
    data['EMA_short'] = ema_short
    data['EMA_long'] = ema_long
    data['MACD'] = macd
    data['MACD_signal'] = macdsignal
    return data

# Risk Yönetimi: Stop-Loss ve Take-Profit
def manage_risk(order, stop_loss_percentage=10, take_profit_percentage=15):
    entry_price = float(order['fills'][0]['price'])
    
    # Stop-loss ve Take-Profit fiyatlarını hesapla
    stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
    take_profit_price = entry_price * (1 + take_profit_percentage / 100)
    
    # Emirleri yerleştir
    stop_loss_order = exchange.futures_create_order(
        symbol=order['symbol'],
        side='SELL' if order['side'] == 'BUY' else 'BUY',
        type='STOP_MARKET',
        stopPrice=stop_loss_price,
        quantity=order['quantity']
    )
    
    take_profit_order = exchange.futures_create_order(
        symbol=order['symbol'],
        side='SELL' if order['side'] == 'BUY' else 'BUY',
        type='LIMIT',
        price=take_profit_price,
        quantity=order['quantity']
    )
    
    print(f"Stop-Loss ve Take-Profit emirleri yerleştirildi: Stop-Loss: {stop_loss_price}, Take-Profit: {take_profit_price}")

# Volatiliteye Dayalı Dinamik Kaldıraç Ayarlama
def adjust_leverage_based_on_volatility(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)  # Son 100 dakikalık veriyi al
    high_prices = [x[2] for x in ohlcv]
    low_prices = [x[3] for x in ohlcv]
    
    atr = np.mean(np.array(high_prices) - np.array(low_prices))  # ATR hesaplama
    volatility = atr / np.mean(high_prices)  # Volatilite oranı
    
    if volatility > 0.02:
        leverage = 125
    else:
        leverage = 50
    
    # Leverage'ı ayarla
    exchange.futures_leverage(symbol=symbol, leverage=leverage)
    print(f"Volatiliteye göre kaldıraç {leverage} olarak ayarlandı.")

# Binance API ile İşlem Yapma
def open_position(symbol, side, amount, leverage=125):
    try:
        # Kaldıraç ayarlama
        exchange.futures_leverage(symbol=symbol, leverage=leverage)
        
        # İşlem açma
        order = exchange.futures_create_order(
            symbol=symbol,
            side=side,  # BUY veya SELL
            type='MARKET',
            quantity=amount
        )
        print(f"Pozisyon açıldı: {side} {symbol} {amount} BTC")
        return order
    except Exception as e:
        print(f"Hata: Pozisyon açılamadı - {e}")
        return None

# Telegram Bildirim Gönderme
def send_telegram_message(message):
    # Telegram API'sini kullanarak mesaj göndermek
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
    requests.get(url)

# Ana Fonksiyon
def main():
    symbol = 'BTC/USDT'
    amount = 0.001  # İşlem miktarı (örneğin 0.001 BTC)
    
    # Veriyi yükle ve işleyelim
    close_data_scaled, scaler = load_and_preprocess_data()
    X, y = create_dataset(close_data_scaled)
    
    # Modeli eğitelim
    model = train_lstm_model(X, y)
    
    # RSI, EMA ve MACD göstergelerini hesaplayalım
    data = pd.read_csv("bitcoin_data.csv")
    data = calculate_indicators(data)
    
    # Volatiliteyi ve kaldıraç oranını ayarlayalım
    adjust_leverage_based_on_volatility(symbol)
    
    # Modelle işlem açalım
    order = open_position(symbol, 'BUY', amount)
    if order:
        # Risk yönetimini uygulayalım
        manage_risk(order)
        send_telegram_message(f"Yeni işlem açıldı: {order}")
    
    time.sleep(60)  # 1 dakika bekleyelim

if __name__ == "__main__":
    main()
