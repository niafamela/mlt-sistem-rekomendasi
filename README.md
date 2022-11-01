# Laporan Proyek Machine Learning Sistem Rekomendasi Movie - Nia Famela Simanjuntak

## Project Overview
Sistem rekomendasi adalah salah satu yang penerapan teknologi *Machine Learning* yang paling sukses dan tersebar luas dalam bisnis. Ini adalah pendekatan penyaringan informasi yang digunakan untuk memprediksi preferensi pengguna tersebut. Area paling populer di mana sistem rekomendasi diterapkan adalah buku, berita, artikel, musik, video, film, dll. Dalam proyek ini saya telah mengusulkan sistem rekomendasi film yang didasarkan pada pendekatan penyaringan kolaboratif yang membuat penggunaan informasi yang diberikan oleh pengguna, menganalisisnya dan kemudian merekomendasikan film yang paling cocok untuk pengguna di waktu itu. Daftar film yang direkomendasikan diurutkan berdasarkan peringkat yang diberikan untuk film-film ini oleh pengguna sebelumnya dan menggunakan berbagai algoritma Machine Learning untuk tujuan proyek ini.

Ini juga membantu pengguna untuk menemukan film pilihan mereka berdasarkan pengalaman film pengguna lain secara efisien dan efektif tanpa membuang banyak waktu dalam penjelajahan yang tidak berguna. Sistem rekomendasi yang disajikan menghasilkan rekomendasi menggunakan berbagai jenis pengetahuan dan data tentang pengguna dari kumpulan data film. Pengguna kemudian dapat menelusuri rekomendasinya dengan mudah dan menemukan film pilihan mereka.

Referensi : [Movie recommendation System](https://d1wqtxts1xzle7.cloudfront.net/64371278/IRJET-V7I4718-with-cover-page-v2.pdf?Expires=1666251395&Signature=VnAqNa-A~UQzIfptRi~xG4esgC5YFfnoSB92ZtnZnWSsROoITUgWnQKWBW5AQqerI0hsfpiOkD54P-9p6CoCPeDyvEQqS3scrKzNjZ9RQFs5tZt4sDgaduKOP32d74ttGpqrYbkFb9ienMvQvCuf6cKQZT352OYyWUQJhZa8KpCePEbTA8sDNpcNu97LOAbT139zRBTpG3ag7CLdr0bfV6fPAuY05poRJRbmP5YuCBSaxLwd8w0lxb9n8QjiyHFyGBOxD1Iihowa1qJN3wDdSlcQidBjKFBMULvoQUBL4DuAFr9HpScs76sQp54qqyGYPYlG5z-SL-xwj6-4M2rPsg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

## Business Understanding

Proyek ini dibangun untuk bisnis dengan membuat suatu rekomendasi movie secara efektif dan efisien

### Problem Statements
Bagaimana cara merekomendasikan movie yang disukai user lain dapat juga direkomendasikan kepada user yang lainnya?

### Goals
Membuat model *Machine Learning* yang dapat merekomendasikan movie secara akurat berdasarkan ratings dan aktivitas pengguna di masa lalu 

### Solution Statements
Solusi yang saya buat dalam membuat proyek sistem rekomendasi movie ini dengan menggunakan 2 algoritma *Machine Learning* ,yaitu :
- *Content Based Filtering* merupakan merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.
- *Collaborative filtering* bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten.

## Data Understanding
Dataset yang digunakan merupakan kumpulan data film pengguna berdasarkan parameter yang berbeda dengan mengambil input sebagai nama film dan memberikan output sebagai saran film bersama dengan skor kesamaan. Dataset ini dapat diunduh pada link berikut [Movies Recommendation System](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data?select=ml-latest-small). Dataset tersebut memiliki format csv dengan 4 fitur cvs yaitu links.csv, movies.csv, ratings.csv dan  tags.csv.
#### Variabel-variabel pada Movies Recommendation System yaitu sebagai berikut:
- links : merupakan daftar link movie tersebut, dimana banyak data = 9742
- movies : merupakan daftar movie yang tersedia, dimana Banyak data = 9742
- ratings : merupakan daftar penilaian yang diberikan pengguna terhadap movie, dimana jumlah data rating = 100836
- tags : merupakan daftar kata kunci dari movie tersebut.

Dimana pada tahapan ini dilakukan Univariate Exploratory Data Analysis dan Data Preprocessing yaitu dengan menggabungkan seluruh movieId pada kategori movies, menggabungkan seluruh userId, mengetahui Jumlah Rating, mencheck missing value, menggabungkan Data dengan fitur tags movies.

## Data Preparation
Yang dilakukan pada tahapan ini yaitu:
- Mengecek missing value pada dataframe all_movies
- Membersihkan missing value dengan fungsi dropna()
- Mengecek kembali missing value pada variabel all_movies_clean
- Mengurutkan movie berdasarkan movieId kemudian memasukkannya ke dalam variabel fix_movies
- Mengecek berapa jumlah data pada fix_movies
- Membuat variabel preparation yang berisi dataframe fix_resto kemudian mengurutkan berdasarkan placeID
- Membuang data duplikat pada variabel preparation
- Mengonversi data series ‘movieId’ , ‘title’ , ‘genres’ menjadi dalam bentuk list ke dalam variabel ‘movies_id’, ‘mobies_title’, dan ‘movies_genre
- Membuat dictionary untuk data ‘movies_id’, ‘mobies_title’, dan ‘movies_genre’

## Modeling dan Result
Pada tahapan ini dalam membuat model, saya lakukan dengan pemodelan algoritma *Machine Learning* yaitu *content based filtering* dan *collabrative filtering*. 
Kedua model tersebut yaitu:
### Model Development dengan Content Based Filtering
Sistem rekomendasi berbasis konten (*content-based filtering*) adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. 
Pertama, cek lagi data yang kita miliki dan assign dataframe dari tahap sebelumnya ke dalam variabel data, dan berikut tampilan dari 5 data pada variabel tersebut:
|      	| id 	          |                movies_title 	|                       movies_genre   	|
|-----	|-------------	|---------------------------- 	|-------------------------------------	|
|  546 	|         2387 	|      Very Bad Things (1998) 	|                       Comedy\|Crime 	|
| 1159 	|         7713 	|           Cat People (1942) 	|    Drama\|Horror\|Romance\|Thriller 	|
|  892 	|         5349 	|           Spider-Man (2002) 	| Action\|Adventure\|Sci-Fi\|Thriller 	|
| 1170 	|         7924 	| Stray Dog (Nora inu) (1949) 	|          Drama\|Film-Noir\|Thriller 	|
| 1090 	|         6981 	|    Ordet (Word, The) (1955) 	|                               Drama 	|

Pada model ini, dilakukan beberapa tahap yaitu:
##### TF-IDF Vectorizer
Pada tahap ini, kita akan membangun sistem rekomendasi sederhana berdasarkan movie_genre yang disediakan berdasarkan movie_title. Pada proyek ini, kita juga menggunakan fungsi tfidfvectorizer() dari library sklearn.
Yang dilakukan pada tahapan ini yaitu:
- Inisialisasi TfidfVectorizer
- Melakukan perhitungan idf pada data movies_genre
- Mapping array dari fitur index integer ke fitur nama
- Melakukan fit lalu ditransformasikan ke bentuk matrix
- Melihat ukuran matrix tfidf
- Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
- Membuat dataframe untuk melihat tf-idf matrix, mengisi kolom dengan genre dan mengisi baris dengan movies_title

##### Cosine Similarity
- Menghitung cosine similarity pada matrix tf-idf
- Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa movies_title
- Melihat similarity matrix pada setiap movies

##### Mendapatkan Rekomendasi
Pada tahapan ini, membuat fungsi movies_recommendations dengan beberapa parameter sebagai berikut:
- title_of_movies : Judul film atau movie (index kemiripan dataframe).
- similarity_data : Dataframe mengenai similarity yang telah kita definisikan sebelumnya.
- items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘movies_title’ dan ‘movies_genre’.
- k : Banyak rekomendasi yang ingin diberikan.

Sehingga didapatkan movie yang memiliki genre yang sama dengan movie tersebut yaitu Comedy dan Drama 'In Good Company (2004)' dan menemukan rekomendasi movies yang mirip dengan movie 'In Good Company (2004)' tersebut dengan menggunakan model *Content Based Filtering* yaitu movie yang disukai pengguna dimasa lalu sebagai berikut:
|   	|                                 movies_title 	| movies_genre  	|
|-- 	|---------------------------------------------	|---------------	|
| 0 	|                         Kolya (Kolja) (1996) 	| Comedy\|Drama 	|
| 1 	|                         Meet John Doe (1941) 	| Comedy\|Drama 	|
| 2 	| Man on the Train (Homme du train, L') (2002) 	| Comedy\|Drama 	|
| 3 	|                            Radio Days (1987) 	| Comedy\|Drama 	|
| 4 	|             Everything Is Illuminated (2005) 	| Comedy\|Drama 	|



### Model Development dengan Collaborative Filtering
Pada tahapan ini akan merekomendasikan item yang mirip dengan preferensi pengguna di masa lalu. Saya akan menerapkan teknik *collaborative filtering* untuk membuat sistem rekomendasi. Teknik ini membutuhkan data rating dari user. 
#### Data Understanding
Melakukan load data di awal dan membaca file dataset ratings.csv. Untuk memudahkan supaya tidak tertukar dengan fitur ‘ratings’ pada data, kita ubah nama variabel ratings menjadi df. Pada data ini terdapat 100836 baris dan 4 kolom.

#### Data Preparation
Pada tahapan ini beberapa hal yang dilakukan yaitu:
- Mengubah userId menjadi list tanpa nilai yang sama
- Melakukan encoding userId
- Melakukan proses encoding angka ke ke userId
- Mengubah movieId menjadi list tanpa nilai yang sama
- Melakukan proses encoding movieId
- Melakukan proses encoding angka ke movieId
- Mapping userId ke dataframe movies_genre
- Mapping movieId ke dataframe movies
- Mendapatkan jumlah user dan jumlah movies
- Mengubah ratings menjadi nilai float
- Nilai minimum dan maksimal ratings
```sh
Number of User: 610, Number of Movies: 9724, Min Rating: 0.5, Max Rating: 5.0
```
#### Membagi Data untuk Training dan Validasi
Pada tahapan ini beberapa hal yang dilakukan yaitu:
- Mengacak dataset
- Membuat variabel x untuk mencocokkan data movies_genre dan movies menjadi satu value
- Membuat variabel y untuk membuat rating dari hasil 
- Membagi menjadi 80% data train dan 20% data validasi

#### Proses Training
- Pada tahap ini, model menghitung skor kecocokan antara pengguna dan movie dengan teknik embedding. 
- Selain itu, juga dapat menambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.
- Di sini, saya membuat class RecommenderNet dengan keras Model class. Selanjutnya, lakukan proses compile terhadap model. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. 

#### Mendapatkan Rekomendasi Movies
- Untuk mendapatkan rekomendasi movie, pertama ambil sampel user secara acak dan definisikan variabel movies_not_watche yang merupakan daftar movie yang belum pernah ditonton oleh pengguna.
- Variabel movies_not_watched diperoleh dengan menggunakan operator bitwise (~) pada variabel movies_watched_by_user
- Untuk memperoleh rekomendasi movies, gunakan fungsi model.predict() dari library Keras

Sehingga diperoleh Sistem rekomendasi dengan *Collaborative Filtering* yaitu sebagai berikut:

| Movies with high ratings from user        	|                                              	|
|-------------------------------------------	|----------------------------------------------	|
| Princess Mononoke (Mononoke-hime) (1997)  	| Action\|Adventure\|Animation\|Drama\|Fantasy 	|
| Donnie Darko (2001)                       	| Drama\|Mystery\|Sci-Fi\|Thriller             	|
| Thor: Ragnarok (2017)                     	| Action\|Adventure\|Sci-Fi                    	|

| Top 10 movies recommendation         	|                                     	|
|--------------------------------------	|-------------------------------------	|
| Last Days of Disco, The (1998)       	| Comedy\|Drama                       	|
| More (1998)                          	| Animation\|Drama\|Sci-Fi\|IMAX      	|
| Crossing Delancey (1988)             	| Comedy\|Romance                     	|
| Lady Jane (1986)                     	| Drama\|Romance                      	|
| Awful Truth, The (1937)              	| Comedy\|Romance                     	|
| Midnight Clear, A (1992)             	| Drama\|War                          	|
| Woman Under the Influence, A (1974)  	| Drama                               	|
| Adam's Rib (1949)                    	| Comedy\|Romance                     	|
| Safety Last!                         	| Action\|Comedy\|Romance             	|
| Into the Woods (1991)                	| Adventure\|Comedy\|Fantasy\|Musical 	|

## Evaluation

#### Hasil Evaluasi untuk Content Based Filtering
Disini saya merekomendasikan film 'In Good Company (2004)'

![1](https://user-images.githubusercontent.com/92345291/196939496-76410d48-f70a-4584-8eb6-ad5003ba4bfb.png)

Hasil dari Top-N 5 dari film atau movie yang saya rekomendasikan adalah sebagai berikut :

![3](https://user-images.githubusercontent.com/92345291/196939300-c0efb311-7cc1-47b4-a809-cc8078c45598.png)

Dari 5 item yang direkomendasikan, 5 item memiliki genre Comedy|Romance (similar). Artinya, precision sistem kita sebesar 5/5 atau 100%.
Teknik Evaluasi di atas adalah dengan menggunakan precission, rumus dari teknik ini adalah:
.
```sh
Precision = #of recommendation that are relevant/#of item we recommend
```

##### Hasil Evaluasi untuk Collaborative Filtering

- Metrik Evaluasi yang digunakan yaitu root mean squared error (RMSE).
RMSE adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar.

- Matriks RMSE memiliki kelebihan dan kekurangan yaitu kelebihannya menghukum kesalahan besar lebih sehingga bisa lebih tepat dalam beberapa kasus. Sedangkan Kekurangannya yaitu memberikan bobot yang relatif tinggi untuk kesalahan besar. Ini berarti RMSE harus lebih berguna ketika kesalahan besar sangat tidak diinginkan.
- Formula dari matriks RMSE adalah sebagai berikut:

![6](https://user-images.githubusercontent.com/92345291/196946781-dc2a7b8b-7aaa-4ef1-a5fa-925382746cbf.png)

```sh
keterangan :
At : Nilai Aktual.
ft = Nilai hasil peramalan.
N = banyaknya dataset
```

Cara menerapkan metrik tersebut adalah dengan menambahkan 'metrics=[tf.keras.metrics.RootMeanSquaredError()]' pada model.compile seperti dibawah ini :

![7](https://user-images.githubusercontent.com/92345291/196968269-c18d6f7f-6a82-4073-b520-5ce8f439ede8.png)

- Hasil dari model evaluasi visualisasi matriks adalah sebagai berikut :

![4](https://user-images.githubusercontent.com/92345291/196940376-00a91738-a926-45c2-ade4-a8da833e97b0.png)

Dari visualisasi data diatas, proses training model cukup smooth dan model konvergen pada epochs sekitar 100. Saya memperoleh nilai error akhir sebesar sekitar 0.19 dan error pada data validasi sebesar 0.20. Dimana Nilai tersebut cukup bagus untuk sistem rekomendasi. 


### References
Surendran, A., Yadav, A. K., & Kumar, A. (2020). Movie recommendation system using machine learning algorithms. Int Res J Eng Technol


