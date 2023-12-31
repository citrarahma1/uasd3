# Laporan Proyek Machine Learning
### Nama : Citra Rahmawati
### Nim :211351037
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Proyek yang saya buat yaitu pengkalkulasian resiko seseorang terkena stroke. Menurut peneliti, Faktor risiko utama terjadinya stroke adalah tekanan darah tinggi. Maka dari itu, saya selaku pembuat mencoba membuat pengkalkulasian apakah seseorang dapat terkena stroke atau tidak dari beberapa atribut yang di dapatkan dari dataset. Agar seseorang dapat lebih waspada dan terhindar dari penyakit stroke.

## Business Understanding
Proyek ini memudahkan serta menghemat waktu kita dalam mengecek apakah kita dapat terkena penyakit stroke atau tidak,dengan cukup mengisi pertanyaan yang ada tanpa harus pergi ke rumahsakit terlebih dulu.

Bagian laporan ini mencakup:

### Problem Statements
- Ketidaktahuan seseorang terhadap dirinya yang dapat terkena penyakit stroke atau tidak.

### Goals
- Untuk mengetahui apakah kita dapat terkena penyakit stroke atau tidak.

    ### Solution statements
    - Dikembangkannya Kalkulasi terkena stroke berbasis web agar dapat mengetahui dengan mudah apakah kita dapat terkena penyakit stroke atau tidak dengan parameter yang telah ditentukan dan dihitung menggunakan algoritma decision tree.

## Data Understanding
Dataset yang digunakan adalah dataset yang diambil dari kaggle, dimana isi dari Brain Stroke Dataset ini yaitu hasil perkiraan seseorang dapat terkena penyakit stroke dan diteliti berdasarkan 10 atribut.

[Brain Stroke Dataset](https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset/data).

### Variabel-variabel pada Brain Stroke Dataset adalah sebagai berikut:
- gender: Merupakan jenis kelamin.
- age: Merupakan umur seseorang.
- hypertension: Merupakan sebuah pertanyaan apakah seseorang memiliki riwayat penyakit darah tinggi.
- heart disease: Merupakan sebuah pertanyaan apakah seseorang memiliki riwayat penyakit jantung.
- Ever-married: Merupakan sebuah pertanyaan apakah seseorang pernah menikah.
- work type: Merupakan jenis pekerjaan.
- Residencetype: Merupakan tipe tempat tinggal.
- avg glucose level: Merupakan rata-rata kadar glukosa.
- BMI: Merupakan indeks masa tubuh.
- smoking_status: Merupakan sebuah pertanyaan apakah seseorang adalah perokok.

## Data Preparation
Pertama import dulu library yang di butuh dengan memasukan perintah :
```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
```
```bash
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

Kemudian agar dataset di dalam kaggle langsung bisa terhubung ke collab maka harus membuat token terlebih dahulu di akun kaggle dengan memasukan perintah :
```bash
from google.colab import files
files.upload()
```

Berikutnya yaitu membuat direktori dengan memasukan perintah :
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Setelah itu kita panggil url dataset yang ada di website kaggle untuk didownload langsung ke google colab.
```bash
!kaggle datasets download -d jillanisofttech/brain-stroke-dataset
```

Selanjutnya kita ekstrak dataset yang sudah didownload dengan perintah :
```bash
!mkdir brain-stroke-dataset
!unzip brain-stroke-dataset.zip -d brain-stroke-dataset
!ls brain-stroke-dataset
```

Jika berhasil diekstrak, maka kita langsung dapat membuka dataset tersebut dengan perintah :
```bash
df = pd.read_csv('/content/brain-stroke-dataset/brain_stroke.csv')
```

Untuk menampilkan isi dari dataset dengan memasukan perintah :
```bash
df.head()
```

Untuk menampilkan jumlah data dan kolom yang ada di dataset, masukan perintah :
```bash
df.info()
```

Untuk menampilkan berapa kolom yang ada, masukan perintah :
```bash
df.shape
```



**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Deployment
pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

