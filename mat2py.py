import subprocess

# List of packages to install
packages_to_install = ['pandas', 'numpy', 'matplotlib', 'tkinter', 'scipy', 'scikit-learn']

for package in packages_to_install:
    try:
        # Attempt to import the package
        __import__(package)
    except ImportError:
        print(f'{package} is not installed. Installing...')
        subprocess.call(['pip', 'install', package])

import os
import pandas as pd  
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal.windows import hann
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
scantime = 25
from sklearn.decomposition import PCA


# Create a tkinter root window and hide it
root = tk.Tk()

# Reading Raw data from a CSV file
file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

if file_path:
    print("Reading logfile", file_path)
    
    # Assuming the CSV file has a header row with column names
    df = pd.read_csv(file_path, skiprows=[0, 1], usecols=range(17))
    scn = pd.read_csv(file_path, skiprows = [0,1, 2])
    df.drop([' ScanData', ' Reserved', ' Reserved.1', ' Reserved.2', ' Reserved.3'], axis=1, inplace=True)
    column_mapping = {
    ' Timestamp': 'T',
    ' MessageId': 'msgID',
    ' SourceId': 'srcID',
    ' EmbeddedTimestamp': 'Tstmp',
    ' ScanStartPs': 'Tstrt',
    ' ScanStopPs': 'Tstp',
    ' ScanStepBins': 'Nbin',
    ' Filtering': 'Nfilt',
    ' AntennaId': 'antID',
    ' Reserved.4': 'Imode',
    ' NumSamplesTotal': 'Nscn',
    }
    df.rename(columns=column_mapping, inplace=True)
    
    root.withdraw()
else:
    print("No file selected")
    root.withdraw()

# scn = scn[scn.columns[0]].str.split(',', expand=True)
header_row = pd.DataFrame([scn.columns], columns=scn.columns)
result_df = pd.concat([header_row, scn])
result_df = result_df.reset_index(drop=True)
result_df = result_df.iloc[:, 16:]
result_df['scn'] = result_df.apply(lambda row: row.tolist(), axis=1)
result_df = result_df.drop(result_df.columns[:-1], axis=1)
scn = pd.concat([df, result_df], axis=1)

# B SCAN
# Find indices where 'Nfilt' is equal to 1 in the 'scn' DataFrame
rawscansI = np.where(scn['Nfilt'] == 1)[0]
# Initialize an empty array for b-scan data
bscan_data = np.zeros((len(rawscansI), len(scn['scn'][0])))
# Populate the b-scan data array
for i in range(len(rawscansI)):
    bscan_data[i, :] = np.array(scn['scn'][rawscansI[i]])

xet = 20
# Calculate the new values for the first row of 'bscan_data'
for i in range(96):
    # Initialize a new matrix
    new_matrix = 0 
    for j in range(xet):
        new_matrix = new_matrix + bscan_data[j][i]
    new_matrix = new_matrix / xet
    bscan_data[1][i] = new_matrix

for i in range(96):
    for j in range(2,71):
        bscan_data[j][i] = bscan_data[j][i] - bscan_data[1][i]


# Dimensions of data and converting matrix
m, n = bscan_data.shape

# Medyan Filtresi ile arkaplan temizliği
window_size = 13  # Pencere boyutu
background_removed_bscan_data = medfilt(bscan_data, kernel_size=(window_size, window_size))

# SVD işleminin uygulanması
U, S, Vt = np.linalg.svd(background_removed_bscan_data, full_matrices=False)
k = 2
X = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Elde edilen matrisin boyutlarının atanması
M = U.shape[0]
N = Vt.shape[1]

# Eigen değerlerin alınması
eigen_values = S**2

# SVD ile filtreleme işleminin düzeltilmesi
sv_filtered = np.zeros((m, n))
for g in range(1, k):
    lmd = S[g] * np.outer(U[:, g], Vt[g, :])
    sv_filtered += lmd

# Grafik Çizimleri
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(bscan_data, cmap='gray')
plt.title('Orjinal B-Scan Data')

plt.subplot(2, 2, 2)
plt.imshow(background_removed_bscan_data, cmap='gray')
plt.title('Arka Plan Gurultusu Çikarilmis B-Scan Data')

plt.subplot(2, 2, 3)
plt.imshow(X, cmap='gray')
plt.title('SVD Sonrasi Dusuk Boyutlu Temsil')

plt.subplot(2, 2, 4)
plt.imshow(sv_filtered, cmap='gray')
plt.title('SVD ile Filtrelenmis Veri')

plt.tight_layout()

# Normal plot grafiğini çizdirme
k_value = 200  # 50
plt.figure(figsize=(10, 4))
plt.plot(np.arange(n), bscan_data[k_value, :], 'b', label='Orjinal')
plt.plot(np.arange(n), background_removed_bscan_data[k_value, :], 'r', label='Arka Plan Gurultusu Cikarilmis')
plt.plot(np.arange(n), sv_filtered[k_value, :], 'g', label='SVD ile Filtrelenmis')
plt.legend()
plt.title(f'{k_value}. Satir Verisi Uzerinde Plot Grafigi')
plt.xlabel('Sutunlar')
plt.ylabel('Degerler')
plt.grid()

plt.show()

# MTI filtresi parametreleri
alpha = 0.9  # Yumuşatma faktörü
threshold = 0.01  # Eşik değeri

mti_filteredSignal = np.zeros(sv_filtered.shape)
mti_filteredSignal[0, :] = sv_filtered[0, :]  # İlk örneği orijinal veri ile başlat

for n in range(1, sv_filtered.shape[0]):
    diff = sv_filtered[n, :] - sv_filtered[n-1, :]
    thresholded_diff = diff * (np.abs(diff) > threshold)  # Eşik değeri kullanarak farkı sıfırla
    mti_filteredSignal[n, :] = (1 - alpha) * mti_filteredSignal[n-1, :] + alpha * thresholded_diff

# Sonuç olarak elde edilen veri
mti_result = mti_filteredSignal

# Grafiklerin çizdirilmesi
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.imshow(sv_filtered, cmap='jet', aspect='auto')
plt.colorbar()
plt.title('SVD ile Filtrelenmiş Veri')

plt.subplot(2, 1, 2)
plt.imshow(mti_result, cmap='jet', aspect='auto')
plt.colorbar()
plt.title('MTI ile Hareketli Nesnelerin Algilanmasi')

# MTI Sonucunun normal plot grafiğini çizdirme
k = 200  # 50
n = sv_filtered.shape[1]  # MTI sonucunun sütun sayısı SVD sonucunun sütun sayısına eşit olmalıdır.
plt.figure(figsize=(10, 4))
plt.plot(np.arange(n), sv_filtered[k, :], 'b', label='SVD ile Filtrelenmiş')
plt.plot(np.arange(n), mti_result[k, :], 'r', label='MTI ile Hareketli Nesnelerin Algilanmasi')
plt.legend()
plt.title(f'Satir {k} Verisi Üzerinde MTI Sonucu Plot Grafiği')
plt.xlabel('Sütunlar')
plt.ylabel('Değerler')
plt.grid()

plt.show()

from sklearn.decomposition import PCA

# Dimensions of data and converting matrix
m, n = bscan_data.shape

# Median Filtering
window_size = 13
background_removed_bscan_data = medfilt(bscan_data, kernel_size=(window_size, window_size))

# PCA for initial filtering
k_pca = 2
pca = PCA(n_components=k_pca)
X_pca = pca.fit_transform(background_removed_bscan_data)
filtered_with_pca = pca.inverse_transform(X_pca)

# SVD for initial filtering
U, S, Vt = np.linalg.svd(filtered_with_pca, full_matrices=False)
k_svd = 2
X_svd = U[:, :k_svd] @ np.diag(S[:k_svd]) @ Vt[:k_svd, :]

# Additional PCA filtering
k_additional_pca = 1  # Choose the number of components for additional filtering
pca_additional = PCA(n_components=k_additional_pca)
X_additional_pca = pca_additional.fit_transform(X_svd)
filtered_with_additional_pca = pca_additional.inverse_transform(X_additional_pca)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(bscan_data, cmap='gray')
plt.title('Original B-Scan Data')

plt.subplot(2, 2, 2)
plt.imshow(filtered_with_pca, cmap='gray')
plt.title('Data Filtered with PCA (First Pass)')

plt.subplot(2, 2, 3)
plt.imshow(X_svd, cmap='gray')
plt.title('Data Filtered with SVD (First Pass)')

plt.subplot(2, 2, 4)
plt.imshow(filtered_with_additional_pca, cmap='gray')
plt.title('Data Filtered with PCA (Second Pass)')

plt.tight_layout()
plt.show()

# Veri boyutlarını alın
m, n = bscan_data.shape
ss = np.zeros((m, 3))
fundamental_distance = np.inf  # Başlangıçta en küçük mesafeyi sonsuz olarak ayarlayın

# Zirve tespiti için kullanılacak parametreler
peak_threshold = 0.2  # Zirve eşiği
min_peak_distance = 100  # Minimum zirve mesafesi

for pl in range(m):
    signal = mti_result[pl, :]

    # Sinyali normalize edin
    signal = signal / np.max(np.abs(signal))

    # Zirveleri ve zirve indekslerini bulun
    indexesOfPeaks, properties = find_peaks(signal, height=peak_threshold, distance=min_peak_distance)
    peaks = properties['peak_heights']
    if len(peaks) > 0:
        # En yüksek zirveyi bulun
        bigindex = indexesOfPeaks[np.argmax(signal[indexesOfPeaks])]
        #bigindex = indexesOfPeaks[maxPeakIndex]
        # Uzaklık Hesabı
        tof = bigindex * 61  # tof=Time of Flight
        distance = tof * (3 * 1e-4) / 2

        # Geçerli bir aralıkta mı kontrol edin
        if 0 < distance < 5:  # Bu aralık isteğe göre ayarlanır
            ss[pl, 2] = distance
            ss[pl, 1] = bigindex
            ss[pl, 0] = pl

            # En küçük mesafeyi güncelle
            if distance < fundamental_distance:
                fundamental_distance = distance

# Mesafeleri kontrol edin ve yalnızca geçerli olanları çizin
valid_distances = ss[ss[:, 2] > 0, :]

plt.figure()
plt.bar(valid_distances[:, 0], valid_distances[:, 2])
plt.xlabel("İndeksler")
plt.ylabel("Mesafe (m)")

# 'peak_threshold', bir zirvenin kabul edilebilir yükseklik eşiğini belirler.
# 'min_peak_distance', ardışık zirveler arasındaki minimum mesafeyi tanımlar.
# Bu algoritma, belirli bir sinyalin içindeki zirveleri tespit eder ve bu zirvelerin zaman farklarına dayalı olarak mesafelerini hesaplar.

# Parametrelerin ayarlanması
k = 200
x = mti_result[:, k]

# Analitik genlik zarfını elde etme
y = x

# Pencereleme uygulama
window = hann(len(y))
y_windowed = y * window

# Sinyalin FFT'sini alma
y_fft = np.fft.fft(y_windowed)

# Frekans eksenini oluşturma
Fs = len(mti_result) / (scantime)
n = len(y_fft)
freq_axis = np.arange(n) * (Fs / n)

# Tek taraflı genlik zarfını hesaplama
y_fft_single_sided = y_fft[:n//2+1]
freq_axis_single_sided = freq_axis[:n//2+1]

# Solunum Frekansı Hesaplama (0.2-0.6 Hz arası)
respiration_range = (freq_axis_single_sided >= 0.2) & (freq_axis_single_sided <= 0.6)
respiration_peak_freqs = freq_axis_single_sided[respiration_range]
peaks, _ = find_peaks(np.abs(y_fft_single_sided[respiration_range]), height=0.2 * np.max(np.abs(y_fft_single_sided[respiration_range])), distance=1)
respiration_freq = respiration_peak_freqs[peaks]

# Düşük geçiren filtre tasarımı
filter_order = 20  # Filtre sırası
cutoff_freq_low = 0.6  # Düşük kesim frekansı (Hz)
cutoff_freq_high = 1  # Yüksek kesim frekansı (Hz)
b, a = butter(filter_order, cutoff_freq_low / (Fs / 2), btype='low')
filtered_signal = filtfilt(b, a, y)

# Analitik genlik zarfını elde etme
y_hilbert = np.abs(hilbert(filtered_signal))

# Kalp Atışı Frekansı Hesaplama (1-1.4 Hz arası)
heart_rate_range = (freq_axis_single_sided >= 1) & (freq_axis_single_sided <= 1.4)
heart_rate_peak_freqs = freq_axis_single_sided[heart_rate_range]
peaks, _ = find_peaks(np.abs(y_fft_single_sided[heart_rate_range]), height=0.2 * np.max(np.abs(y_fft_single_sided[heart_rate_range])), distance=1)
heart_rate_freq = heart_rate_peak_freqs[peaks]

# Genlik Yüksekliği Olan Peak'leri Yumuşatma (0.6-1 Hz arası)
noise_range = (freq_axis_single_sided > 0.6) & (freq_axis_single_sided <= 1)
smoothed_fft = np.abs(y_fft_single_sided)
smoothed_fft[noise_range] = np.convolve(smoothed_fft[noise_range], np.ones(10)/10, mode='same')

# Daha fazla genlik yumuşatma (0.6-1 Hz arası)
num_smooth_passes = 4  # Yumuşatma işlemi sayısı isteğe göre seçilir
for i in range(num_smooth_passes):
    smoothed_fft[noise_range] = np.convolve(smoothed_fft[noise_range], np.ones(10)/10, mode='same')

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(y)), -y)
plt.title('Analitik Genlik Zarfi')
plt.xlabel('Zaman (örnek)')
plt.ylabel('Genlik')
plt.grid()

# 0-5 Hz aralığını bulma
desired_freq_range = (freq_axis_single_sided >= 0) & (freq_axis_single_sided <= 5)
desired_freq_indices = np.where(desired_freq_range)[0]

# Genlik değerlerini en yüksek peak'in genlik değerine bölmek
max_genlik_degeri = np.max(smoothed_fft[desired_freq_indices])
smoothed_fft[desired_freq_indices] = smoothed_fft[desired_freq_indices] / max_genlik_degeri

plt.subplot(2, 1, 2)

# Sadece 0-5 Hz aralığını çizdirme
plt.plot(freq_axis_single_sided[desired_freq_indices], smoothed_fft[desired_freq_indices], 'k')

plt.title('Tek Tarafli Analitik Genlik Zarfinin Frekans Spektrumu (0-5 Hz)')
plt.xlabel('Frekans (Hz)')
plt.ylabel('Genlik')
plt.grid()
plt.legend(['0-5 Hz', 'Solunum Frekansi', 'Kalp Atisi Frekansi'])

print(f'Solunum Frekansi: {respiration_freq} Hz')
print(f'Kalp Atisi Frekansi: {heart_rate_freq} Hz')
print(f'Mesafe: {fundamental_distance} m')

plt.tight_layout()
plt.show()


# for i in respiration_freq:
#     if i >= 0.30 and i <= 0.5:
#         fundamental_freq1 = i
#     else:
#         fundamental_freq1 = respiration_freq[0]  # Solunum frekansını kullan
# for c in heart_rate_freq:
#     if c >= 1.19 and c <= 1.30:
#         fundamental_freq2 = c
#     else:
#         fundamental_freq2 = heart_rate_freq[0]

for i in respiration_freq:
    if i >= 0.39 and i <= 0.5:
        fundamental_freq1 = i
for c in heart_rate_freq:
    if c >= 1.19 and c <= 1.30:
        fundamental_freq2 = c


# fundamental_freq1 = respiration_freq[0]  # Solunum frekansını kullan
# fundamental_freq2 = heart_rate_freq[0]   # Kalp atışı frekansını kullan

fundamental_distance = distance    # Burada fundamental frekansın kesiştiği mesafe değerini belirtin

# Yatay eksendeki mesafe aralığı
min_distance = 0    # Minimum mesafe
max_distance = 3    # Maksimum mesafe
num_distances = mti_result.shape[1]  # Mesafe verisi sayısı

# Yatay eksendeki mesafe verilerini oluşturma
distance_values = np.linspace(min_distance, max_distance, num_distances)

# Temel frekansların bulunduğu indeksleri hesaplama
fundamental_index1 = np.where(np.abs(freq_axis_single_sided - fundamental_freq1) <= 1)[0]
fundamental_index2 = np.where(np.abs(freq_axis_single_sided - fundamental_freq2) <= 1)[0]

# Mesafe etrafında yoğunluk haritasını oluşturma
density_map = np.zeros((len(freq_axis_single_sided), num_distances))

# Yalnızca temel frekanslarda ve belirli frekans aralığındaki bileşenlerde yoğunluğu hesapla
for i in range(len(freq_axis_single_sided)):
    if (i in fundamental_index1 or i in fundamental_index2):
        if 0.6 <= freq_axis_single_sided[i] <= 1:
            density_map[i, int(np.round(interp1d(distance_values, np.arange(num_distances))(fundamental_distance)))] = 0
        else:   
            density_map[i, int(np.round(interp1d(distance_values, np.arange(num_distances))(fundamental_distance)))] = np.abs(y_fft_single_sided[i])

# Yoğunluğu normalize etme
normalized_density_map = density_map / np.max(density_map)
normalized_density_map = normalized_density_map**2

# Kontur haritasını görselleştirme
plt.figure()
# Yatay ekseni özelleştirerek belirtilen aralıkta gösterme
min_display_distance = fundamental_distance - 0.5
max_display_distance = fundamental_distance + 0.5
display_distance_indices = np.where((distance_values >= min_display_distance) & (distance_values <= max_display_distance))[0]

# Yatay ekseni özelleştirerek fundamental frekans aralığında gösterme
min_display_frequency = min(fundamental_freq1, fundamental_freq2) - 0.5
max_display_frequency = max(fundamental_freq1, fundamental_freq2) + 0.5
display_frequency_indices = np.where((freq_axis_single_sided >= min_display_frequency) & (freq_axis_single_sided <= max_display_frequency))[0]

plt.contourf(freq_axis_single_sided[display_frequency_indices], distance_values[display_distance_indices],
             (normalized_density_map[display_frequency_indices][:, display_distance_indices]).T, 20, cmap='jet', linestyles='none')

text = ' Breath'
text2 = 'Pulse'
plt.colorbar()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Distance (m)')
plt.title('')
plt.text(0.30, fundamental_distance - 0.08, text,fontsize = 12, color = '#ffffff')
plt.text(1.15, fundamental_distance - 0.08, text2,fontsize = 12, color = '#ffffff')
# En yoğun noktanın koordinat bilgilerini yazdırma
print('En Yoğun Nokta Koordinatlari:')
print(f'Solunum Frekansi: {fundamental_freq1} Hz')
print(f'Kalp Atisi Frekansi: {fundamental_freq2} Hz')

plt.show()
