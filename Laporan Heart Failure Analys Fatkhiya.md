# Laporan Akhir Kecerdasan Buatan - Kelompok 11

**Anggota Kelompok :**

1. Fatkhiya Firdausy Nuzulia Suryonegoro - NIM. 3.34.21.2.09
2. Muhammad Rizqi Vicky - NIM. 3.34.21.2.16

## Domain Proyek

Gagal ginjal merupakan suatu kondisi medis yang serius dan kompleks di mana fungsi ginjal untuk menyaring dan membersihkan darah dari limbah dan racun menurun secara signifikan. Penyakit ini dapat berakibat fatal jika tidak dikelola dengan baik. Gagal ginjal dapat mempengaruhi berbagai aspek kehidupan pasien, termasuk kualitas hidup, harapan hidup, serta beban finansial akibat biaya perawatan.

Proyek ini bertujuan untuk menganalisis dataset pasien dengan gagal ginjal dan mengidentifikasi faktor-faktor yang berkontribusi pada risiko kematian. Dataset ini berisi atribut klinis dan demografis pasien, termasuk usia, anemia, ejection_fraction, tekanan darah tinggi, dan lain-lain. Analisis akan dilakukan menggunakan teknik Machine Learning untuk memprediksi kematian serta menemukan hubungan antara variabel dengan kematian pada pasien gagal ginjal.


## Business Understanding

### Problem Statements

Berdasarkan permasalahan yang diuraikan sebelumnya, *problem statements* dari proyek kali ini adalah sebagai berikut.
- Apa saja faktor-faktor yang berpengaruh terhadap kematian pada pasien dengan gagal ginjal?
- Bagaimana penerapan teknik Machine Learning dalam membangun model prediksi risiko kematian pada pasien gagal ginjal yang dapat membantu dalam pengambilan keputusan medis yang lebih baik?

### Goals

Adapun tujuan yang hendak dicapai dari proyek ini adalah sebagai berikut.
- Melakukan <em> preprocessing </em> data sehingga data tersebut siap untuk di latih oleh model <em> Machine Learning </em> untuk memprediksi faktor apa saja yang mempengaruhi kematian pasien gagal ginjal.
- Membandingkan beberapa algoritma *Machine Learning* menggunakan prosedur Kfold Cross Validation. Setelah dilakukan perbandingan, selanjutnya dilakukan prediksi dengan menghitung confusion matrix dan akurasi dari model yang nantinya akan mencetak grafik ROC Curve yang menunjukkan hubungan antara True Positive Rate dan False Positive Rate.

### **Solution Statements**

Adapun solusi yang saya ajukan untuk menyelesaikan permasalah adalah sebagai berikut.

1. Melakukan analisis deskriptif untuk mengetahui pola dan informasi yang tersimpan di data mengenai fitur yang mempengaruhi 
2. Melakukan  *Preprocessing* data seperti cek *missing value*, membedakan anatara data numerik dan kategorikal, melakukan evaluasi terhadap tiap fitur yang ada dalam data, serta memisahkan data menjadi dua bagian.
3. Membuat model **klasifikasi** dengan mencoba 5 algoritma, yaitu  **LogisticRegression**, **DecisionTree**, **Random Forest**, **AdaBoost**, dan **SVC**.

### **Implementation**

Setelah mencapai tujuan di atas, hasil analisis dan model terbaik dapat diimplementasikan untuk memberikan rekomendasi atau informasi yang berguna bagi manajemen dan perawatan pasien dengan gagal ginjal. Dengan memahami faktor-faktor risiko dan korelasi antara variabel dengan kematian pada pasien, langkah-langkah penanganan dan perawatan yang lebih tepat dapat diambil. Pemerintah dapat berkolaborasi dengan pihak-pihak terkait, seperti rumah sakit, dokter, dan peneliti, untuk menerapkan rekomendasi dari proyek analisis data gagal ginjal ini. Hal ini diharapkan dapat meningkatkan kualitas hidup pasien, mengurangi risiko kematian, serta mengoptimalkan sumber daya dan pengelolaan dalam penanganan kasus gagal ginjal.

## Data Understanding
Data yang digunakan adalah dataset yang berasal dari kaggle. Data ini berisikan berbagai atribut prediksi kematian akibat heart failure (gagal ginjal). Dataset dapat diakses pada tautan berikut [Heart Failure Prediction](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

#### Variabel yang ada pada Heart Failure Prediction Datasets sebagai berikut:
1. **age:** Usia pasien yang menderita gagal ginjal.
2. **anaemia:** Variabel biner (0 atau 1) yang menunjukkan apakah pasien menderita anemia (1) atau tidak (0).
3. **creatinine_phosphokinase:** Tingkat enzim kreatinin fosfokinase dalam darah pasien.
4. **diabetes:** Variabel biner yang menunjukkan apakah pasien menderita diabetes (1) atau tidak (0).
5. **ejection_fraction:** Persentase fraksi darah yang dipompa keluar dari ventrikel kiri saat detak jantung.
6. **high_blood_pressure:** Variabel biner yang menunjukkan apakah pasien memiliki tekanan darah tinggi (1) atau tidak (0).
7. **platelets:** Jumlah trombosit dalam darah pasien.
8. **serum_creatinine:** Tingkat kreatinin dalam darah pasien.
9. **serum_sodium:** Tingkat natrium dalam darah pasien.
10. **sex:** Jenis kelamin pasien (pria=1 atau wanita=0).
11. **smoking:** Variabel biner yang menunjukkan apakah pasien merokok (1) atau tidak (0).
12. **time:** Waktu pengamatan atau follow-up pasien dalam periode tertentu.
13. **DEATH_EVENT:** Variabel biner yang menunjukkan apakah pasien mengalami kematian (1) atau tidak (0). Atribut ini menjadi atribut target dalam analisis, di mana prediksi akan dilakukan untuk mengetahui faktor-faktor yang berkontribusi terhadap kematian pasien gagal ginjal.

### *Visualization*

Visualisasi persentase variabel kategorikal pada kelompok data yang memiliki nilai DEATH_EVENT = 1. Proses ini bertujuan untuk mengetahui distribusi persentase dari setiap variabel kategorikal (anaemia, diabetes, high_blood_pressure, sex, dan smoking) pada kondisi pasien yang mengalami "DEATH_EVENT"

![Gambar 1](https://s2.loli.net/2023/07/23/7MT15QYPh8of46z.png)                                     					Gambar 1. Pie Chart Data vs DEATH_EVENT



Berikut adalah perbedaan nilai rata-rata (mean) dari variabel numerik pada kelompok data yang memiliki "DEATH_EVENT" = 1 (Death Event Occurred) dan "DEATH_EVENT" = 0 (No Death Event Occurred).

![Gambar 2](https://s2.loli.net/2023/07/23/DkIFo8WUOlTxp2a.png)

​															Gambar 2. Perbedaan Mean berdasarkan nilai DEATH_EVENT




## Data Preparation

Pada tahap ini dilakukan beberapa proses sebagai berikut.

1. **Handling Missing Values**: Proses penanganan nilai yang hilang adalah langkah untuk mengatasi data yang tidak lengkap atau terdapat nilai yang kosong (NA/null). Penanganan missing values dapat dilakukan dengan mengisi nilai yang hilang dengan nilai lain, menghapus baris atau kolom yang mengandung nilai hilang, atau menggunakan teknik imputasi berdasarkan statistik dari data yang tersedia.
2. **Differentiating Numerical and Categorical Data**: Proses ini melibatkan pengklasifikasian kolom-kolom dalam dataset menjadi dua kelompok berdasarkan tipe datanya. Data numerik adalah data berupa angka atau bilangan, sementara data kategorikal adalah data yang menggambarkan kategori atau kelompok.
3. **Dropping Columns**: Proses ini melibatkan penghapusan kolom tertentu dari dataset. Penghapusan kolom dilakukan ketika kolom tersebut dianggap tidak relevan atau kurang berkontribusi dalam analisis atau pemodelan data.
4. **Feature Analysis**: Proses ini melibatkan evaluasi dan pemahaman lebih lanjut terhadap setiap fitur dalam dataset. Melalui analisis fitur, kita dapat mengetahui pentingnya masing-masing fitur dalam mempengaruhi hasil prediksi atau target. Fitur yang memiliki pengaruh tinggi dapat dipilih untuk digunakan dalam pemodelan atau analisis lebih lanjut.
5. **Confusion Matrix**: mengevaluasi performa model klasifikasi dengan membandingkan hasil prediksi model dengan nilai sebenarnya pada data uji. Matriks confusion berisi empat nilai: true positive (TP), true negative (TN), false positive (FP), dan false negative (FN).
6. **Train-Test Split**: Proses ini melibatkan pemisahan dataset menjadi dua bagian: data latih (train) dan data uji (test). Data latih digunakan untuk melatih model machine learning, sementara data uji digunakan untuk menguji performa model yang telah dilatih.
7. **Data Splitting**: Proses pemisahan data adalah langkah untuk membagi dataset menjadi beberapa bagian atau subset, biasanya untuk keperluan analisis atau pemodelan. Pemisahan data dapat dilakukan berdasarkan persentase tertentu, seperti train-test split, atau dengan teknik lain seperti cross-validation. Tujuannya adalah untuk menguji model pada data yang belum pernah dilihat sebelumnya untuk menghindari overfitting dan mendapatkan estimasi kinerja model yang lebih akurat.

## Modeling
Pada tahap ini dilakukan proses pelatihan untuk mendapatkan model dengan performa terbaik. Tahapan yang dilakukan pada proses Modelling adalah sebagai berikut.

1. **Evaluasi performa model machine learning** 

   Sebelum modeling, dilakukan proses evaluasi performa berbagai model machine learning dengan menggunakan metode validasi silang (cross-validation) dengan Stratified K-Fold terlebih dahulu.

2. **Membandingkan performa masing-masing model**

   Langkah selanjutnya adalah membandingkan performa masing-masing model berdasarkan akurasi, matriks confusion, dan kurva ROC. Hal ini akan membantu dalam menentukan model terbaik untuk digunakan dalam analisis data gagal ginjal. Grafik area di bawah kurva ROC juga memberikan gambaran tentang seberapa baik model dapat membedakan kelas target yang berbeda.

3. **Membuat bar plot perbandingan akurasi**

   Terakhir, membandingkan akurasi dari beberapa model klasifikasi yang telah diuji pada data pengujian (test set). 

#### Models

Pada analisis ini, menggunakan algoritma LogisticRegression, DecisionTree, Random Forest, AdaBoost, dan SVC untuk memprediksi DEATH_EVENT. 

* **LogisticRegression**

  Logistic Regression merupakan salah satu model klasifikasi yang digunakan untuk memprediksi probabilitas kelas target. Model ini cocok digunakan ketika variabel target adalah biner atau dua kelas. Logistic Regression menggabungkan fungsi logistik (sigmoid) untuk menghitung probabilitas dan menggunakan batas keputusan untuk mengklasifikasikan data ke salah satu dari dua kelas yang mungkin.

* **DecisionTree**

  Model klasifikasi yang berbentuk seperti pohon keputusan, di mana setiap node dalam pohon merepresentasikan keputusan berdasarkan fitur-fitur tertentu. Setiap cabang dari node menggambarkan kemungkinan nilai dari fitur tersebut, dan setiap daun pohon merepresentasikan kelas target atau hasil akhir dari keputusan. Decision Tree cocok digunakan untuk masalah klasifikasi dengan fitur-fitur yang dapat dibagi menjadi kategori atau nilai diskrit.

* **Random Forest**

  Random Forest adalah model klasifikasi yang merupakan ensemble learning, di mana sejumlah besar decision tree dikombinasikan untuk membuat prediksi akhir. Setiap tree dibuat dengan menggunakan bagian acak dari data dan fitur-fitur yang dipilih secara acak. Akhirnya, prediksi dari setiap tree digabungkan melalui voting atau rata-rata untuk menghasilkan prediksi akhir. Random Forest memiliki performa yang baik untuk masalah klasifikasi yang kompleks dan besar.

* **AdaBoost**

  Model klasifikasi ensemble learning yang menggabungkan beberapa weak learners (model lemah) menjadi satu model yang lebih kuat. Weak learners adalah model yang memiliki performa sedikit di atas kebetulan atau model yang memiliki akurasi rendah. Dalam setiap iterasi, AdaBoost memberikan bobot lebih pada data yang salah klasifikasi sebelumnya, sehingga iterasi selanjutnya berfokus pada data yang sulit diprediksi. Dengan cara ini, AdaBoost meningkatkan performa model secara bertahap.

* **SVC**

  Model klasifikasi yang digunakan untuk memisahkan data ke dalam kelas-kelas yang berbeda dengan menggunakan hyperplane (bidang pemisah) yang mengoptimalkan margin (jarak) antara kelas. SVC cocok digunakan untuk masalah klasifikasi dengan fitur-fitur yang kontinu dan dapat digunakan untuk kasus klasifikasi dengan lebih dari dua kelas. SVC juga dapat menangani data yang tidak linier dengan menggunakan kernel untuk mentransformasi data ke dimensi yang lebih tinggi.

## Evaluation

Berikut merupakan hasil evaluasi kinerja dari beberapa model klasifikasi yang telah diuji pada data validasi (validation accuracy) dan data pelatihan (training accuracy). 

1. Logistic Regression:

   ![1](https://s2.loli.net/2023/07/23/ufhK6dACiMTHlg7.png)

   ​													Gambar 3.  Hasil evaluasi kinerja dari Logistic Regression

   - Akurasi pada data validasi: 0.7692

   - Akurasi pada data pelatihan: 0.7313

     

2. Decision Tree Classifier:

   ![2](https://s2.loli.net/2023/07/23/giBcM8yPUrAYb1o.png)

   ​													Gambar 4. Decision Tree Classifier

   - Akurasi pada data validasi: 0.6731

   - Akurasi pada data pelatihan: 1.0

     

3. Random Forest Classifier:

   ![3](https://s2.loli.net/2023/07/23/9vaP87o2R6ijNMU.png)

   ​													Gambar 5. Hasil evaluasi kinerja dari Random Forest Classifier

   - Akurasi pada data validasi: 0.8013

   - Akurasi pada data pelatihan: 1.0

     

4. Ada Boost:

   ![4](https://s2.loli.net/2023/07/23/2OgFK4SXwi1GfDW.png)

   ​												Gambar 6. Hasil evaluasi kinerja dari Ada Boost

   - Akurasi pada data validasi: 0.7372

   - Akurasi pada data pelatihan: 0.8449

     

5. SVM (Support Vector Machine):

   ![5](https://s2.loli.net/2023/07/23/o9R6Ypxk3zmULO7.png)

   ​												Gambar 7. Hasil evaluasi kinerja dari SVM

   * Akurasi pada data validasi: 0.7051 

   * Akurasi pada data pelatihan: 0.6759

     


### Final Report

Berikut hasil evaluasi akhir dari 5 Model.![hasil](https://s2.loli.net/2023/07/23/JqP61XBsbFIUnSg.png)

​				Gambar 8. Visualisasi akurasi dari berbagai model klasifikasi pada data uji (test set).

Berdasarkan hasil di atas, Random Forest Classifier adalah model terbaik yang direkomendasikan untuk analisis data gagal ginjal karena memiliki akurasi tertinggi pada data uji.

## Daftar Referensi
Referensi

Azar, Ahmad Taher, et al. "A random forest classifier for lymph diseases." *Computer methods and programs in biomedicine* 113.2 (2014): 465-473. [A random forest classifier for lymph diseases - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169260713003751)

Charbuty, Bahzad, and Adnan Abdulazeez. "Classification based on decision tree algorithm for machine learning." *Journal of Applied Science and Technology Trends* 2.01 (2021): 20-28. https://doi.org/10.38094/jastt20165

Jakkula, Vikramaditya. "Tutorial on support vector machine (svm)." *School of EECS, Washington State University* 37.2.5 (2006): 3. [SVMTutorial (neu.edu)](https://course.ccs.neu.edu/cs5100f11/resources/jakkula.pdf)

Ogunseye, Elizabeth Oluyemisi, et al. "Predictive analysis of mental health conditions using AdaBoost algorithm." *ParadigmPlus* 3.2 (2022): 11-26. [Predictive Analysis of Mental Health Conditions Using AdaBoost Algorithm | ParadigmPlus (itiud.org)](http://journals.itiud.org/index.php/paradigmplus/article/view/37)

Nusinovici, Simon, et al. "Logistic regression was as good as machine learning for predicting major chronic diseases." *Journal of clinical epidemiology* 122 (2020): 56-69. https://doi.org/10.1016/j.jclinepi.2020.03.002

Widodo, Slamet, Herlambang Brawijaya, and Samudi Samudi. "Stratified K-fold cross validation optimization on machine learning for prediction." *Sinkron: jurnal dan penelitian teknik informatika* 7.4 (2022): 2407-2414. [Stratified K-fold cross validation optimization on machine learning for prediction | Sinkron : jurnal dan penelitian teknik informatika (polgan.ac.id)](https://polgan.ac.id/jurnal/index.php/sinkron/article/view/11792)