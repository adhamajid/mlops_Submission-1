# Submission 1: Movie Reviews Sentiment Classification

Nama: Adha Syah Majid

Username Dicoding: [asyah_majid](https://www.dicoding.com/users/asyah_majid/)

|                             | Deskripsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**                 | [Complaint Sample](https://www.kaggle.com/datasets/mohammadripan/complaint-sample)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Latar Belakang**          | Data yang saya gunakan dalam penelitian ini adalah koleksi sampel keluhan dari konsumen yang telah diklasifikasikan untuk tujuan analisis jenis keluhan terhadap produk. Setiap keluhan ditujukan kepada berbagai layanan atau produk tertentu. Dataset ini disusun untuk keperluan klasifikasi multi-kelas, dengan fokus pada empat kategori utama: Mortgage (Hipotek), Debt collection (Penagihan Hutang), Student loan (Pinjaman Pelajar), dan Credit reporting, credit repair services, or other personal consumer reports (Laporan Kredit, Layanan Perbaikan Kredit, atau Laporan Konsumen Pribadi Lainnya). |
| **Solusi Machine Learning** | Untuk menangani tantangan dalam klasifikasi keluhan pelanggan, kami telah mengembangkan solusi berbasis machine learning yang mampu memahami dan mengkategorikan teks keluhan ke dalam empat kategori utama: Mortgage, Debt collection, Student loan, dan Credit reporting, credit repair services, or other personal consumer reports. Pendekatan kami melibatkan penggunaan teknik pemrosesan Natural Language Processing (NLP), yang memungkinkan model untuk mengeksplorasi fitur dan nuansa bahasa yang terdapat dalam teks keluhan tersebut.                                                                |
| **Metode Pengolahan Data**  | Dalam proses pengolahan ini, kami melaksanakan serangkaian langkah. Mulai dari unduhan libraries dan dataset yang diperlukan hingga pembuatan komponen 'Transform'.                                                                                                                                                                                                                                                                                                                                                                                                                                               |

Dalam proyek ini, kami mengembangkan pipeline machine learning menggunakan TensorFlow Extended (TFX), sebuah platform end-to-end dari Google yang dirancang untuk menyederhanakan proses pembuatan, pelatihan, dan penerapan model machine learning secara besar-besaran. TFX membantu dalam otomatisasi dan pemantauan pipeline ML dalam produksi, meningkatkan efisiensi, dan memastikan konsistensi model.

Langkah pertama dalam pipeline kami adalah ExampleGen, yang bertugas memuat dataset dan membaginya menjadi dua bagian: pelatihan dan evaluasi. Pembagian ini dilakukan dengan perbandingan 80:20, memastikan bahwa model memiliki data yang cukup untuk pembelajaran sambil menyimpan sebagian data untuk evaluasi. Ini adalah langkah penting untuk memastikan bahwa model dapat menggeneralisasi dengan baik pada data yang belum pernah dilihat.

Selanjutnya, SchemaGen digunakan untuk menentukan skema dari dataset berdasarkan statistik yang dihasilkan.

ExampleValidator berperan dalam memeriksa dataset untuk mencari potensi anomali atau inkonsistensi berdasarkan statistik dan skema yang dihasilkan.

Dalam modul Transform, kami melakukan operasi pra-pemrosesan sederhana pada dataset dengan mengubah semua teks keluhan ('Consumer_complaint') menjadi huruf kecil. Langkah ini membantu mengurangi kompleksitas model dengan menghilangkan perbedaan antara huruf besar dan kecil. Meskipun pra-pemrosesan ini sederhana, kami tetap terbuka untuk menjalankan proses pra-pemrosesan yang lebih kompleks jika diperlukan, tergantung pada kinerja model nantinya.

Setelah menyiapkan data dan menjalankan pra-pemrosesan dengan Transform, kami memasukkan komponen Tuner untuk meningkatkan performa model dengan mencari konfigurasi hiperparameter terbaik secara otomatis. Ini dilakukan menggunakan Keras Tuner, sebuah library yang memfasilitasi penyetelan hiperparameter untuk model TensorFlow. |
| **Arsitektur Model** | Arsitektur model dibangun di dalam modul trainer dengan melibatkan langkah-langkah berikut:

Input dan Preprocessing Teks: Kami menggunakan TextVectorization untuk standarisasi teks dan mengonversinya menjadi bilangan bulat. Layer ini disesuaikan dengan teks keluhan untuk membangun vocabulari yang diperlukan untuk model.

Embedding: Layer embedding mengubah bilangan bulat dari langkah sebelumnya menjadi vektor padat, memungkinkan model untuk mempelajari representasi yang kaya dari teks keluhan.

Global Average Pooling 1D: Digunakan untuk mereduksi dimensi output dari lapisan embedding.

Dense Layers: Sejumlah lapisan Dense ditambahkan berdasarkan hyperparameter num_hidden_layers. Setiap lapisan memiliki 32 unit dan menggunakan fungsi aktivasi ReLU.

Output Layer: Layer terakhir adalah lapisan dense dengan fungsi aktivasi softmax, menghasilkan distribusi probabilitas di antara kelas-kelas yang ada.

Hyperparameter seperti jumlah lapisan tersembunyi ditentukan oleh hasil dari proses tuning. Setelah mempersiapkan data dan melakukan pra-proses dengan Transform, kami memasukkan komponen Tuner untuk meningkatkan performa model dengan mencari konfigurasi hyperparameter terbaik secara otomatis. Proses ini dilakukan menggunakan Keras Tuner, sebuah library yang memfasilitasi penyetelan hiperparameter untuk model TensorFlow.

Untuk pelatihan, kami menggunakan modul trainer. Di dalamnya, kami membangun dan melatih model untuk klasifikasi keluhan pelanggan dengan memanfaatkan TensorFlow Transform untuk pra-pemrosesan data dan TensorFlow untuk pembangunan model. Model ini dirancang untuk memahami teks keluhan konsumen dan mengklasifikasikannya ke dalam satu dari empat kategori utama. |
| **Metrik Evaluasi** | Dalam proyek ini, kita akan menggunakan TensorFlow Model Analysis (`TFMA`) untuk mengevaluasi model yang telah saya latih. TFMA memungkinkan evaluasi model secara komprehensif dengan menggunakan berbagai metrik. Konfigurasi evaluasi yang saya tetapkan adalah sebagai berikut:

1. **ExampleCount**: Menghitung jumlah sampel yang dievaluasi, memberikan gambaran tentang ukuran dataset evaluasi.

2. **SparseCategoricalCrossentropy**: Mengukur seberapa baik prediksi model dibandingkan dengan label sebenarnya, menggunakan crossentropy yang merupakan metrik umum untuk masalah klasifikasi multi-klas.

3. **SparseCategoricalAccuracy**: Mengukur akurasi model, yaitu persentase prediksi yang tepat dari semua prediksi yang dibuat.

4. **Precision (top_k=1)**: Menghitung presisi, atau proporsi prediksi positif yang benar, dengan fokus pada prediksi teratas (top 1).

5. **Recall (top_k=1)**: Menghitung recall, atau proporsi positif yang benar teridentifikasi, lagi-lagi dengan fokus pada prediksi teratas (top 1).

6. **MultiClassConfusionMatrixPlot**: Menyediakan visualisasi confusion matrix untuk analisis lebih lanjut tentang bagaimana model melakukan prediksi di antara kelas yang berbeda.

7. **SparseCategoricalAccuracy**: Evaluasi ini diakhiri dengan metrik akurasi kategorikal yang jarang, dengan threshold khusus yang ditetapkan untuk memastikan bahwa model mencapai akurasi minimal 50%, dan perubahan positif yang signifikan didefinisikan sebagai perbaikan minimal 0.0001. |
   | **Performa Model** | Dalam mengevaluasi performa model yang kami kembangkan, kami menemukan bahwa model mencapai akurasi pelatihan yang tinggi, mencapai **97.67%**, sementara akurasi pada data validasi adalah **85.59%**. Baik nilai precision maupun recall pada data validasi adalah **85.693%**, menunjukkan bahwa model ini memiliki kemampuan yang baik dalam mengklasifikasikan keluhan ke dalam kategori yang benar secara keseluruhan.

Namun, terdapat perbedaan yang signifikan antara akurasi pelatihan dan validasi, menandakan adanya kemungkinan _overfitting_. _Overfitting_ terjadi ketika model terlalu mempelajari detail dan noise pada data latih hingga kehilangan kemampuan untuk generalisasi pada data yang belum pernah dilihat sebelumnya.

Meskipun demikian, model ini masih bermanfaat sebagai asisten AI untuk membantu dalam klasifikasi awal keluhan pelanggan. Tujuan utama model ini adalah untuk mengurangi beban kerja manual dengan memberikan prediksi awal yang kemudian akan ditinjau dan diperiksa oleh manusia. Dalam konteks ini, model memberikan dasar yang kuat untuk analisis lebih lanjut, namun tidak diharapkan untuk menggantikan sepenuhnya proses pengambilan keputusan manusia. |