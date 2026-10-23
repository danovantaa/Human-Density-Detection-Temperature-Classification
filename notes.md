# Definisi variabel :

- c1 = Channel Input
- c2 = Channel Output
- e = expansion ratio (yang akan dikalikan dengan output untuk hidden channel pada bottleneck)
- k = kernel
- s = stride
- p = padding
- act = Aktivasi
- c\_ = hidden channel
- g = Groups.
- d = Dilation.
- dim=1 : dalam channel

**dilation** = Memperluas area tangkapan (receptive field) tanpa menambah jumlah parameter.
**Receptive field** = area gambar yang dilihat neuron.

# Definisi variabel :

- Autopad = hitung padding otomatis agar ukuran output konvolusi = ukuran input (same padding).
- nn.Identity()= hanya meneruskan input menjadi output tanpa perubahan apa pun.

# YOLO V8 :

## Backbone

### Convolution

**Fungsi Kegunaan :** untuk mengekstraksi fitur dari gambar

**Struktur:**

```
Conv2d -> BatchNorm2d -> SiLU
```

**Penjelasan:**

- **Conv2d** = Mengambil fitur dasar seperti garis, tepi, dan sudut.
- **BatchNorm2d** = Menormalkan nilai fitur dalam 1 batch agar training lebih stabil.
  - affine = punya parameter γ dan β (bisa belajar)
  - track_running_stats = simpan mean & var global untuk inference stabil
  - eps = nilai kecil untuk mencegah pembagian nol saat normalisasi.
  - momentum = update running mean/var secara lambat agar stabil.

- **SiLU (Sigmoid Linear Unit)** = Membuat model belajar pola non-linear dengan lebih halus dan stabil.

### Bottleneck

**Konsep :** Menggunakan hidden channel (c\_ = e × c2) untuk memproses fitur sebelum diproyeksikan kembali ke jumlah channel output sehingga komputasi menjadi lebih efisien.

**Struktur:**

```
Convolution -> Convolution -> Shortcut
```

**Penjelasan:**

- Menggunakan dua convolution kecil untuk memproses fitur.
- Shortcut (residual connection) menjaga informasi penting agar tidak hilang.

### C2f

**Konsep :** Membagi feature map menjadi dua bagian, kemudian salah satu bagian diproses secara bertahap oleh beberapa Bottleneck. Seluruh feature map hasil setiap tahap digabungkan kembali sehingga representasi fitur menjadi lebih kaya dengan biaya komputasi yang tetap efisien.

**Struktur:**

```
Convolution
      │
    Split
      │
 ┌────┴────┐
 │         │
 │   Bottleneck
 │         │
 │   Bottleneck
 │         │
 │      ...
 └────Concat
        │
   Convolution
```

**Penjelasan:**

- **Split** = Feature map dibagi menjadi dua bagian channel. Satu bagian dipertahankan sebagai jalur langsung, sedangkan bagian lainnya diproses secara bertahap oleh beberapa Bottleneck.

- **Bottleneck** = Fitur dipadatkan dan diproses dengan dua convolution kecil.

### Concat

**Konsep**
Menggabungkan beberapa feature map pada dimensi channel sehingga informasi dari berbagai jalur dapat digunakan secara bersamaan.

**Struktur**

```
Feature1
Feature2
  ...
   ↓
Concat(dim=1)
```

### SPPF (Spatial Pyramid Pooling Fast)

**Konsep**

- Menangkap informasi konteks dari berbagai skala objek melalui pooling bertingkat secara sangat efisien, sehingga model dapat memahami objek kecil, sedang, dan besar secara bersamaan.

**Struktur :**

```
Convolution -> MaxPool2d -> MaxPool2d -> MaxPool2d -> Concat -> Convolution
```

**Penjelasan :**

- **MaxPool2d** = Mengambil nilai terbesar dari area kecil,  
  sehingga memperkuat kemampuan model dalam mendeteksi objek dengan berbagai ukuran.

## Neck

**Konsep**

- Menggabungkan feature map dari berbagai skala menggunakan jalur top-down dan bottom-up sehingga informasi semantik dan detail spasial dapat dipertahankan secara bersamaan.

**Komponen**

- Upsample
- Concat
- C2f
- Downsample

**Alur**

```
Top-Down (FPN)
      ↓
Bottom-Up (PAN)
```

## Head

**Struktur :**

```
Regression Branch       Classification Branch
      Conv                        Conv
        ↓                           ↓
      Conv                        Conv
        ↓                           ↓
  Conv2d(4×reg_max)             Conv2d(nc)
```

### DFL

**Penjelasan :** DFL tidak memprediksi koordinat bounding box secara langsung, tetapi memprediksi distribusi probabilitas untuk setiap sisi bounding box, kemudian menghitung nilai kontinu menggunakan nilai ekspektasi (expected value).

**Struktur :**

```
input  : (B, 4·reg_max, H, W)
reshape: (B, 4, reg_max, H, W)
softmax: distribusi bin
expect : Σ(p·i)
output : (B, 4, H, W)
```

# YOLO V11

## Backbone

### C3K

**Konsep :** Mengembangkan bottleneck pada YOLOv8 dengan menggunakan beberapa bottleneck berukuran kernel 3×3 secara berurutan sehingga ekstraksi fitur menjadi lebih dalam tanpa meningkatkan kompleksitas secara signifikan

- **Struktur:**

```
  Convolution -> Bottleneck*n -> Concat -> Convolution
```

**Penjelasan:**

- Input diproyeksikan menjadi dua jalur.
- Jalur pertama diproses oleh beberapa Bottleneck berturut-turut.
- Jalur kedua menjadi shortcut untuk mempertahankan informasi awal.
- Kedua jalur digabung (Concat), kemudian diproyeksikan kembali menggunakan convolution.

### C3K2

**Konsep :** Merupakan pengembangan dari C2f pada YOLOv8 yang menggunakan blok C3K sebagai ekstraktor fitur utama. Tujuannya adalah meningkatkan kedalaman representasi fitur tanpa membuat jumlah cabang sebanyak C2f sehingga komputasi tetap efisien.

- **Struktur:**
  ```
  Convolution
        │
      Split
    ┌───────┐
    │       │
  Shortcut  C3K × n
    │       │
    └──Concat
          │
    Convolution
  ```
- c3k=False = ekstraksi fitur sederhana dan cepat
- c3k=True = ekstraksi fitur lebih dalam dan kuat

**Penjelasan:**

- Feature map terlebih dahulu diproyeksikan menggunakan convolution.
- Hasilnya dibagi menjadi dua bagian channel.
- Salah satu bagian diproses oleh beberapa blok C3K.
- Bagian lainnya menjadi shortcut.
- Seluruh fitur digabung menggunakan concat.
- Convolution terakhir menggabungkan seluruh informasi menjadi feature map keluaran

### Attention

**Konsep :** Memberikan perhatian (attention) terhadap area dan channel yang penting sehingga fitur yang relevan diperkuat sebelum diproses lebih lanjut.

- **Struktur:**

```
              Convolution
                  │
              Split (Q,K)
                  │
        Element-wise Multiply
                  │
                Sigmoid
                  │
 Element-wise Multiply (Input × Attention)
                  │
Depthwise Convolution (Positional Encoding)
                  │
              Convolution
```

**Penjelasan :**

- Input diproyeksikan menjadi dua feature map (Q dan K).
- Kedua feature map dikalikan untuk menghasilkan hubungan antar fitur.
- Fungsi sigmoid menghasilkan attention map dengan nilai antara 0–1.
- Attention map digunakan untuk memperkuat fitur penting pada input.
- Depthwise convolution menambahkan informasi posisi (positional encoding).
- Convolution terakhir memproyeksikan kembali ke jumlah channel semula.

### FFN (Feed Forward Network)

**Konsep :** Meningkatkan kemampuan representasi fitur setelah attention dengan memperluas jumlah channel kemudian mengompresnya kembali.

- **Struktur:**

```
Convolution
      │
Convolution
```

**Penjelasan :**

- Convolution pertama memperbesar jumlah channel.
- Aktivasi SiLU meningkatkan kemampuan representasi non-linear.
- Convolution kedua mengembalikan jumlah channel ke ukuran semula.

### PSABlock

**Konsep :** Menggabungkan Attention dan Feed Forward Network (FFN) dengan residual connection sehingga model dapat menangkap hubungan global tanpa menghilangkan informasi awal.

- **Struktur:**

```
  Attention
     │
Residual Add
     │
    FFN
     │
Residual Add
```

**Penjelasan :**

- Input diproses menggunakan Attention.
- Hasil Attention dijumlahkan dengan input (Residual Connection).
- Selanjutnya diproses menggunakan FFN.
- Hasil FFN kembali dijumlahkan dengan feature map sebelumnya.
- Residual connection membantu menjaga stabilitas gradien selama pelatihan.

### C2PSA

**Konsep :** Mengintegrasikan PSABlock ke dalam struktur C2 sehingga hanya sebagian channel yang diproses menggunakan attention. Pendekatan ini memberikan kemampuan menangkap konteks global dengan biaya komputasi yang tetap rendah.

- **Struktur:**

```
Convolution
      │
    Split
   ┌─────────┐
   │         │
PSABlock    Shortcut
   │         │
   └──Concat
        │
   Convolution
```

**Penjelasan :**

- Feature map diproyeksikan menggunakan convolution.
- Channel dibagi menjadi dua bagian.
- Salah satu bagian diproses oleh beberapa PSABlock.
- Bagian lainnya dilewatkan langsung sebagai shortcut.
- Kedua jalur digabung menggunakan concat.
- Convolution terakhir menghasilkan feature map keluaran.

## Neck

Pada YOLOv11, struktur PAN-FPN masih digunakan seperti pada YOLOv8, namun blok C2f diganti menjadi C3K2 pada seluruh proses fusi fitur.

```python
YOLOv8      YOLOv11
Upsample    Upsample
   │          │
Concat      Concat
   │          │
  C2f        C3K2
```

**penjelasan :**

- mekanisme top-down dan bottom-up tetap sama.
- Upsample dan Concat tetap digunakan untuk menggabungkan feature map dari berbagai skala.
- Perbedaannya terletak pada penggunaan C3K2 yang menghasilkan representasi fitur lebih dalam dibanding C2f.

## Head

### Depthwise Separable Classification Head

**Konsep :** Branch klasifikasi pada YOLOv11 menggunakan Depthwise Separable Convolution untuk mengurangi jumlah parameter dan komputasi dibanding convolution biasa yang digunakan pada YOLOv8.

- **Struktur:**

```
  Bounding Box Branch
Conv → Conv → Conv2d(4×reg_max)

Classification Branch
  DWConv → Conv
        │
  DWConv → Conv
        │
  Conv2d(nc)
```

**Penjelasan :**

- Branch regresi bounding box tetap sama seperti YOLOv8.
- Branch klasifikasi menggunakan dua blok Depthwise Convolution + Pointwise Convolution.
- Pendekatan ini mempertahankan akurasi dengan jumlah parameter yang lebih sedikit.

### DWConv (Depthwise Convolution)

**Konsep :** Melakukan konvolusi pada setiap channel secara terpisah sehingga komputasi menjadi jauh lebih ringan dibanding convolution standar.

- **Struktur:**

```
Depthwise Convolution
        │
    BatchNorm
        │
      SiLU
```

**Penjelasan :**

- Setiap channel diproses menggunakan kernel tersendiri.
- Tidak terjadi pencampuran antar channel pada tahap ini.
- Pencampuran informasi antar channel dilakukan oleh convolution 1×1 (Pointwise Convolution) setelahnya.
- Mengurangi FLOPs dan jumlah parameter secara signifikan

# YOLO V12 :

## Backbone

### Attention (Self-Attention)

**Konsep :**

- Pada YOLOv11 attention hanya menggunakan Q dan K untuk menghasilkan attention map secara ringan (sigmoid(q × k)), sedangkan YOLOv12 menggunakan mekanisme Self-Attention lengkap yang melibatkan Query (Q), Key (K), dan Value (V).

- Mempelajari hubungan antar posisi pada seluruh feature map menggunakan mekanisme Self-Attention sehingga model mampu menangkap konteks global dengan lebih baik

- **Struktur:**

```
  Convolution
      │
 Split (Q, K, V)
      │
    Q × Kᵀ
      │
    Softmax
      │
  Attention × V
      │
  Convolution
      │
+ Positional Encoding
```

**Penjelasan :**

- Q (Query) merepresentasikan fitur yang ingin mencari informasi.
- K (Key) merepresentasikan fitur yang akan dibandingkan.
- V (Value) menyimpan informasi yang akan diteruskan.
- Perkalian Q × Kᵀ menghasilkan hubungan antar posisi pada feature map.
- Softmax mengubah hubungan tersebut menjadi bobot perhatian (attention weight).
- Bobot attention digunakan untuk menggabungkan informasi dari Value.
- Positional Encoding melalui Depthwise Convolution 7×7 menambahkan informasi - posisi sehingga attention tetap memahami lokasi objek.

### ConvPE (Convolution Positional Encoding)

**Konsep :** Menambahkan informasi posisi menggunakan Depthwise Convolution sehingga Self-Attention tetap mengetahui letak spasial setiap fitur.

- **Struktur:**

```
Depthwise Conv(7×7)
      │
BatchNorm
      │
    SiLU
```

**Penjelasan :**

- Self-Attention cenderung kehilangan informasi posisi.
- ConvPE memperkenalkan kembali informasi spasial menggunakan convolution 7×7 secara depthwise.
- Berbeda dengan positional embedding pada Vision Transformer, ConvPE tidak memerlukan parameter posisi terpisah.

### ABlock

**Konsep :** Menggabungkan Self-Attention dan MLP dalam satu blok residual sehingga model mampu mempelajari hubungan global sekaligus memperkaya representasi kanal.

- **Struktur:**

```
Self-Attention
      │
Residual Add
      │
      MLP
      │
Residual Add
```

**Penjelasan :**

- Input terlebih dahulu diproses menggunakan Self-Attention.
- Hasilnya dijumlahkan dengan input awal (Residual Connection).
- Selanjutnya diproses oleh MLP untuk memperkaya representasi fitur.
- Residual kedua menjaga stabilitas pelatihan.

### MLP

**Konsep :** Memperluas jumlah channel kemudian mengompresnya kembali untuk meningkatkan kemampuan representasi fitur setelah proses attention.

- **Struktur:**

```
Convolution
      │
Convolution
```

**Penjelasan :**

- Convolution pertama memperbesar jumlah channel (hidden = 1.2 × channel).
- Convolution kedua mengembalikan jumlah channel ke ukuran semula.

### A2C2f (Residual ELAN / R-ELAN)

**Konsep :** Mengembangkan struktur C2f dengan menggabungkan beberapa Attention Block secara bertingkat menggunakan konsep Residual ELAN (R-ELAN) sehingga representasi fitur menjadi lebih dalam namun tetap stabil.

- **Struktur:**

```
Convolution
      │
    ABlock
      │
    ABlock
      │
     ...
      │
    Concat
      │
Convolution
      │
Scaled Residual
```

**Penjelasan :**

- Feature map diproyeksikan menggunakan convolution.
- Feature diproses secara bertahap oleh beberapa ABlock.
- Seluruh keluaran setiap tahap digabung menggunakan concat.
- Convolution akhir menggabungkan seluruh representasi fitur.
- Hasil akhirnya ditambahkan ke shortcut menggunakan residual berskala (γ) sehingga kontribusi residual dapat dipelajari selama training.

### Scaled Residual (γ)

**Konsep :** Mengontrol seberapa besar kontribusi residual terhadap output menggunakan parameter yang dapat dipelajari.

**Struktur:**

```
Output = Shortcut + γ × Feature
```

**Penjelasan :**

- γ merupakan parameter yang dipelajari selama training.
- Model dapat menentukan sendiri seberapa besar informasi residual digunakan.
- Membantu menjaga stabilitas ketika jaringan menjadi semakin dalam.

## Neck

**Konsep :** Secara keseluruhan Neck YOLOv12 masih menggunakan PAN-FPN, tetapi blok yang digunakan berubah.

**Perbedaan:**

```
YOLOv8  YOLOv11   YOLOv12
  C2f    C3K2      A2C2f
```

**Penjelasan :**

- Jalur FPN menggunakan A2C2f dengan blok C3K.
- Jalur PAN masih menggunakan A2C2f.
- Bagian akhir menggunakan C3K2 yang dimodifikasi khusus untuk Neck.

# Temperature Cls Head

**Konsep :** Mengklasifikasikan kondisi suhu berdasarkan feature map hasil backbone/neck dengan merangkum informasi spasial menjadi sebuah vektor fitur, kemudian memetakan vektor tersebut ke kelas suhu menggunakan fully connected layer.

**Struktur:**

```
  Feature Map
      │
  Conv(1×1)
      │
  BatchNorm
      │
    SiLU
      │
AdaptiveAvgPool2d
      │
    Flatten
      │
    Dropout
      │
    Linear
      │
Temperature Class
```

**Penjelasan :**

- Conv (1×1) = Mengubah jumlah channel feature map menjadi hidden_channels tanpa mengubah ukuran spasial, sehingga fitur lebih sesuai untuk proses klasifikasi.
- BatchNorm2d = Menormalkan feature map agar distribusi aktivasi lebih stabil selama pelatihan.
- SiLU = Menambahkan non-linearitas sehingga model mampu mempelajari hubungan fitur yang lebih kompleks.
- AdaptiveAvgPool2d(20×20) = Merangkum informasi spasial dengan mengubah ukuran feature map menjadi 20 × 20, tanpa bergantung pada ukuran input awal.
- Flatten = Mengubah feature map 3 dimensi (C × H × W) menjadi sebuah vektor 1 dimensi agar dapat diproses oleh layer fully connected.
- Dropout = Menonaktifkan sebagian neuron secara acak selama pelatihan untuk mengurangi overfitting dan meningkatkan kemampuan generalisasi model.
- Linear (Fully Connected Layer) = Memetakan vektor fitur menjadi skor untuk setiap kelas temperatur.
