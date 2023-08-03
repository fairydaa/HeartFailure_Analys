# HeartFailure_Analys
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
