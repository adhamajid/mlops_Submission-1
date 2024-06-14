# Submission 1: Complaint Classification Pipeline

Nama: Mohammad Ripan Saiful Mansur

Username Dicoding: mohrip16

# Dataset

Dataset: Sampled Complaint Dataset

Link: https://www.kaggle.com/datasets/mohammadripan/complaint-sample

Dataset ini merupakan kumpulan sampel keluhan konsumen yang dikategorikan untuk kebutuhan klasifikasi jenis komplain produk. Terdiri dari teks keluhan yang dialamatkan ke berbagai layanan atau produk, dataset ini dirancang untuk masalah klasifikasi multi-class, dengan empat kategori utama: Mortgage (Hipotek), Debt collection (Penagihan Hutang), Student loan (Pinjaman Pelajar), dan Credit reporting, credit repair services, or other personal consumer reports (Laporan Kredit, Perbaikan Kredit, atau Laporan Konsumen Pribadi Lainnya).

*Catatan: Sebenarnya dataset ini merupakan penulis sendiri yang menyediakan, dikarenakan jika menggunakan dataset full (bukan sample) tfx akan berjalan dengan lambat. Oleh karena itu, penulis mengambil sample sebanyak 20000 jumlah dataset.*

# Masalah

Perusahaan seringkali menghadapi tantangan dalam mengklasifikasikan keluhan pelanggan yang masuk ke berbagai kategori, terutama karena volume keluhan yang besar dan variasi tema keluhan yang luas. Kesulitan ini tidak hanya memperlambat proses respons terhadap pelanggan tetapi juga dapat mengakibatkan penanganan keluhan yang tidak tepat. Dengan demikian, perlu ada sistem yang dapat secara otomatis dan akurat mengklasifikasikan keluhan pelanggan ke dalam kategori yang relevan untuk mempercepat proses penanganan dan meningkatkan kepuasan pelanggan.

# Solusi Machine Learning

Untuk mengatasi masalah klasifikasi keluhan pelanggan, kami mengembangkan sebuah solusi berbasis machine learning yang mampu memahami dan mengklasifikasikan teks keluhan ke dalam empat kategori utama: Mortgage, Debt collection, Student loan, dan Credit reporting, credit repair services, or other personal consumer reports. Pendekatan penulis menggunakan teknik pemrosesan Natural Language Processing atau yang biasa disebut dengan NLP yang memungkinkan model untuk mempelajari fitur dan nuansa bahasa dari teks keluhan.

# Metode pengolahan

Terdapat beberapa hal yang kita lakukan dalam metode pengolahan. Dimulai hingga download libraries dan dataset yang diperlukan hingga pembuatan *component* `Transform`.

Dalam proyek ini, kita mengembangkan pipeline machine learning menggunakan TensorFlow Extended (TFX), sebuah platform end-to-end dari Google yang dirancang untuk menyederhanakan proses pembuatan, pelatihan, dan penerapan model machine learning secara skala besar. TFX membantu otomatisasi dan pemantauan pipeline ML dalam produksi, meningkatkan efisiensi dan memastikan konsistensi model.

Tahap pertama dalam pipeline kita adalah ExampleGen, yang bertugas memuat dataset dan membaginya menjadi dua bagian: training dan evaluasi. Pembagian ini dilakukan dengan perbandingan 80:20, memastikan bahwa model memiliki cukup data untuk belajar sambil tetap memegang sebagian data untuk evaluasi. Ini merupakan langkah penting untuk memastikan bahwa model dapat generalisasi dengan baik pada data yang belum pernah dilihat.

Kemudian, SchemaGen digunakan untuk menentukan skema dari dataset berdasarkan statistik yang dihasilkan.

ExampleValidator berperan dalam memeriksa dataset untuk mencari potensi anomali atau inkonsistensi berdasarkan statistik dan skema yang dihasilkan.

Dalam modul Transform, kita akan melakukan operasi preprocessing sederhana pada dataset dengan mengubah semua teks keluhan (`Consumer_complaint`) menjadi huruf kecil. Langkah ini membantu dalam mengurangi kompleksitas model dengan mengeliminasi perbedaan antara huruf kapital dan non-kapital. Meski preprocessing ini cukup sederhana, kita tetap terbuka untuk melakukan proses preprocessing yang lebih kompleks jika diperlukan, tergantung pada performa model nantinya.

Setelah mempersiapkan data dan melakukan pra-proses dengan Transform, kita akan memasukkan komponen Tuner untuk meningkatkan performa model dengan mencari konfigurasi hyperparameter terbaik secara otomatis. Ini dilakukan menggunakan Keras Tuner, sebuah library yang memfasilitasi hyperparameter tuning untuk model TensorFlow.

# Arsitektur Model

Arsitektur model dibuat di dalam modul trainer. Model dibangun dengan arsitektur yang melibatkan langkah-langkah berikut:

1. **Input dan Preprocessing Teks**: Menggunakan `TextVectorization` untuk standardisasi teks dan konversi menjadi integer. Layer ini disesuaikan dengan teks keluhan untuk membangun vocabulari yang diperlukan untuk model.

2. **Embedding**: Layer embedding mengubah integer dari langkah sebelumnya menjadi vektor dense, memungkinkan model untuk mempelajari representasi yang kaya dari teks keluhan.

3. **Global Average Pooling 1D**: Digunakan untuk mereduksi dimensi output dari embedding layer.

4. **Dense Layers**: Sejumlah lapisan Dense ditambahkan berdasarkan hyperparameter num_hidden_layers. Setiap lapisan memiliki 32 unit dan fungsi aktivasi ReLU.

5. **Output Layer**: Layer terakhir adalah dense layer dengan softmax activation function, menghasilkan distribusi probabilitas di antara kelas-kelas yang ada.

Hyperparameter seperti jumlah lapisan tersembunyi ditentukan oleh hasil dari proses tuning. Setelah mempersiapkan data dan melakukan pra-proses dengan Transform, kita akan memasukkan komponen Tuner untuk meningkatkan performa model dengan mencari konfigurasi hyperparameter terbaik secara otomatis. Ini dilakukan menggunakan Keras Tuner, sebuah library yang memfasilitasi hyperparameter tuning untuk model TensorFlow.

Untuk training sendiri, kita menggunakan modul trainer. Dalam modul trainer, kita mengkonstruksi dan melatih model untuk klasifikasi keluhan pelanggan dengan memanfaatkan TensorFlow Transform untuk pra-pengolahan data dan TensorFlow untuk pembangunan model. Model ini dirancang untuk memahami teks keluhan konsumen dan mengklasifikasikannya ke dalam satu dari empat kategori utama.

# Metrik Evaluasi

Dalam proyek ini, kita akan menggunakan TensorFlow Model Analysis (`TFMA`) untuk mengevaluasi model yang telah saya latih. TFMA memungkinkan evaluasi model secara komprehensif dengan menggunakan berbagai metrik. Konfigurasi evaluasi yang saya tetapkan adalah sebagai berikut:

1. **ExampleCount**: Menghitung jumlah sampel yang dievaluasi, memberikan gambaran tentang ukuran dataset evaluasi.

2. **SparseCategoricalCrossentropy**: Mengukur seberapa baik prediksi model dibandingkan dengan label sebenarnya, menggunakan crossentropy yang merupakan metrik umum untuk masalah klasifikasi multi-klas.

3. **SparseCategoricalAccuracy**: Mengukur akurasi model, yaitu persentase prediksi yang tepat dari semua prediksi yang dibuat.

4. **Precision (top_k=1)**: Menghitung presisi, atau proporsi prediksi positif yang benar, dengan fokus pada prediksi teratas (top 1).

5. **Recall (top_k=1)**: Menghitung recall, atau proporsi positif yang benar teridentifikasi, lagi-lagi dengan fokus pada prediksi teratas (top 1).

6. **MultiClassConfusionMatrixPlot**: Menyediakan visualisasi confusion matrix untuk analisis lebih lanjut tentang bagaimana model melakukan prediksi di antara kelas yang berbeda.

7. **SparseCategoricalAccuracy**: Evaluasi ini diakhiri dengan metrik akurasi kategorikal yang jarang, dengan threshold khusus yang ditetapkan untuk memastikan bahwa model mencapai akurasi minimal 50%, dan perubahan positif yang signifikan didefinisikan sebagai perbaikan minimal 0.0001.

# Performa Model

Dalam evaluasi performa model yang kita kembangkan, kita mendapati bahwa model mencapai akurasi pelatihan yang tinggi, yaitu **97.67%,** sementara akurasi pada data validasi adalah **85.59%**. Kedua nilai precision dan recall pada data validasi adalah **85.693%**, menunjukkan bahwa model ini memiliki kemampuan yang baik dalam mengklasifikasikan keluhan ke dalam kategori yang benar secara keseluruhan.

Namun, terdapat perbedaan yang signifikan antara akurasi pelatihan dan validasi, yang menunjukkan *overfitting*. *Overfitting* terjadi ketika model terlalu baik dalam mempelajari detail dan noise pada data latih hingga kehilangan kemampuan untuk generalisasi pada data yang belum pernah dilihat sebelumnya.

Meskipun demikian, model ini masih cukup baik digunakan sebagai asisten AI untuk membantu dalam klasifikasi awal keluhan pelanggan. Tujuan utama dari model ini adalah untuk mengurangi beban kerja manual dengan menyediakan prediksi awal yang kemudian akan ditinjau dan diperiksa oleh manusia. Dalam kasus ini, model memberikan titik awal yang solid untuk analisis lebih lanjut, dan tidak diharapkan untuk sempurna menggantikan pengambilan keputusan manusia sepenuhnya.