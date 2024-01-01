# Laporan Proyek Machine Learning
### Nama : Citra Rahmawati
### Nim : 211351037
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Proyek yang saya buat yaitu pengkalkulasian resiko seseorang terkena stroke. Menurut peneliti, Faktor risiko utama terjadinya stroke adalah tekanan darah tinggi. Maka dari itu, saya selaku pembuat mencoba membuat pengkalkulasian apakah seseorang dapat terkena resiko stroke atau tidak dari beberapa atribut yang di dapatkan dari dataset. Agar seseorang dapat lebih waspada dan terhindar dari penyakit stroke.

## Business Understanding
Proyek ini memudahkan serta menghemat waktu kita dalam mengecek apakah kita dapat terkena resiko penyakit stroke atau tidak,dengan cukup mengisi pertanyaan yang ada tanpa harus pergi ke rumahsakit terlebih dulu.

Bagian laporan ini mencakup:

### Problem Statements
- Ketidaktahuan seseorang terhadap dirinya yang dapat terkena penyakit stroke atau tidak.

### Goals
- Untuk mengetahui apakah kita dapat terkena penyakit stroke atau tidak.

    ### Solution statements
    - Dikembangkannya Kalkulasi resiko terkena stroke berbasis web agar dapat mengetahui dengan mudah apakah kita dapat terkena penyakit stroke atau tidak dengan parameter yang telah ditentukan dan dihitung menggunakan algoritma decision tree.

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

Untuk pengelompokan data :
```bash
df[df['stroke']==1].groupby(df['gender']).count()
```

Untuk mengubah variabel kategorikal menjadi bentuk biner :
```bash
df['ever_married'] = [ 1 if i !='Yes' else 0 for i in df['ever_married'] ]
df['gender'] = [1 if i != 'Female' else 0 for i in df['gender']]
```

Untuk split data. dan membuat kolom baru :
```bash
df=pd.get_dummies(df,columns=['work_type','Residence_type','smoking_status'])
```

## Modeling
Untuk menentukan x (feature) dan y (label) :
```bash
X=df.drop(['stroke'],axis=1)
y=df['stroke']
```

Untuk memisahkan data training dan data testing dengan memasukan perintah :
```bash
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
```

Lalu masukan data training dan testing ke dalam model decision tree dengan perintah :
```bash
dtc = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0,
    random_state=42, splitter='best'
)

model = dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))
```

Untuk mengecek akurasinya masukan perintah :
```bash
print(f"akurasi data training = {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"akurasi data testing = {dtc_acc} \n")
```
## EDA
untuk menampilkan Distribusi Kasus Stroke Berdasarkan Gender
```bash
plt.subplot(1,2,1)

stroke_cases = df[df['stroke'] == 1]


stroke_counts_by_gender = stroke_cases.groupby('gender').size()


plt.bar(stroke_counts_by_gender.index, stroke_counts_by_gender.values)


plt.xlabel('Gender')
plt.ylabel('Number of Stroke Cases')
plt.title('Distribution of Stroke Cases by Gender')


plt.show()
```
![image](https://github.com/citrarahma1/uasd3/assets/149367504/2c6f3d8d-b124-4038-a4d5-dc299cbc9b7d)

untuk menampilkan Sebaran Kasus Stroke menurut ever_married
```bash
plt.subplot(1,2,1)

stroke_cases = df[df['stroke'] == 1]


stroke_counts_by_ever_married = stroke_cases.groupby('ever_married').size()


plt.bar(stroke_counts_by_ever_married.index, stroke_counts_by_ever_married.values)


plt.xlabel('ever_married')
plt.ylabel('Number of Stroke Cases')
plt.title('Distribution of Stroke Cases by ever_married')


plt.show()
```
![image](https://github.com/citrarahma1/uasd3/assets/149367504/cfa82b0a-4fef-4a2d-a2a3-5a32f2f975a8)


untuk menampilkan Distribusi Kasus Stroke berdasarkan work_type
```bash

plt.subplot(1,2,1)

stroke_cases = df[df['stroke'] == 1]


stroke_counts_by_work_type = stroke_cases.groupby('work_type').size()


plt.bar(stroke_counts_by_work_type.index, stroke_counts_by_work_type.values)


plt.xlabel('work_type')
plt.ylabel('Number of Stroke Cases')
plt.title('Distribution of Stroke Cases by work_type')

plt.xticks(rotation=45)
plt.show()
```
![image](https://github.com/citrarahma1/uasd3/assets/149367504/75799bd7-6c00-44b8-b1df-3192139bc2db)


## Evaluation
Metrik evaluasi yang digunakan yaitu confusion matrik dengan memasukan perintah :
```bash
confusion_mat = confusion_matrix(y_test, y_pred)
```
```bash
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Reds", xticklabels=dtc.classes_, yticklabels=dtc.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Pred')
plt.ylabel('Actual')
plt.show()
```
![image](https://github.com/citrarahma1/uasd3/assets/149367504/c6f6a3f9-82a9-4b86-a914-c7d146f159f6)

## Visualisasi data
untuk memvisualisasikannya :
```bash
ind_col = [col for col in df.columns if col!= 'stroke']
dep_col = 'stroke'
```
```bash
fig = plt.figure(figsize=(20,20))
_ = tree.plot_tree(model,
                   feature_names=ind_col,
                   class_names=dep_col,
                   filled=True)
```
![image](https://github.com/citrarahma1/uasd3/assets/149367504/8c964908-871d-4cd7-94f1-626c7d195f2b)

## Deployment
[Kalkulasi Resiko Terkena Stroke](https://uasdecisiontree.streamlit.app/).
![image](https://github.com/citrarahma1/uasd3/assets/149367504/bf983388-fd78-4d81-937b-60e409b063d8)


