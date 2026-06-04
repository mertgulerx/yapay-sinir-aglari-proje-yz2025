# BLM 3510 Yapay Zeka Proje Ödevi

Bu proje kapsamında 25x25 boyutunda binary matrisler üzerinde çalışan CNN tabanlı yapay sinir ağı modelleri geliştirilmiştir. Projede beş farklı problem için veri setleri oluşturulmuş, modeller eğitilmiş, hiperparametre optimizasyonu yapılmış ve test sonuçları değerlendirilmiştir.

## Proje Konusu

Projenin amacı, yapay sinir ağlarının farklı tipteki geometrik ve sayısal problemleri ne ölçüde öğrenebildiğini incelemektir.

Giriş verisi olarak 25x25 boyutunda binary matrisler kullanılmıştır. Matristeki `1` değerleri noktaları, `0` değerleri ise boş alanları temsil etmektedir.

Çalışmada aşağıdaki problemler ele alınmıştır:

| Problem | Açıklama                                                                                                 | Problem Tipi  |
| ------- | -------------------------------------------------------------------------------------------------------- | ------------- |
| A       | 5 nokta arasındaki en yakın iki noktanın Manhattan mesafesi                                              | Regresyon     |
| B       | 5 nokta arasındaki en uzak iki noktanın Manhattan mesafesi                                               | Regresyon     |
| C       | Matristeki nokta sayısının tahmini                                                                       | Sınıflandırma |
| D       | Nokta sayısının tek veya çift olduğunun tahmini                                                          | Sınıflandırma |
| E       | Nokta sayısı tek ise sol üst köşeye, çift ise sağ alt köşeye en yakın noktanın köşeye Manhattan mesafesi | Regresyon     |

## Veri Üretimi

Veriler C dilinde yazılan program ile üretilmiştir. Her problem için 25x25 binary matrisler oluşturulmuş ve ilgili problem çıktısı hesaplanarak CSV dosyalarına kaydedilmiştir.

Her veri satırı şu yapıdadır:

```text
{matris, çıktı}
```

### Problem A ve B

Problem A ve B için her matrise 5 adet nokta rastgele yerleştirilmiştir. Aynı konuma birden fazla nokta yerleştirilmemiştir.

Problem A için en yakın iki nokta arasındaki Manhattan mesafesi, Problem B için ise en uzak iki nokta arasındaki Manhattan mesafesi hesaplanmıştır.

Manhattan mesafesi şu şekilde hesaplanır:

```text
|x1 - x2| + |y1 - y2|
```

Bu problemler için 10 milyon adet rastgele veri üretilmiştir. Ancak bazı uç değerlerin oluşma ihtimali çok düşük olduğu için tüm teorik aralıklarda eşit sayıda örnek elde edilememiştir. Bu nedenle veri setleri belirli aralıklara filtrelenmiş ve normalize edilmiştir.

| Problem | Değer Aralığı | Aralık Başına Örnek | Toplam Örnek |
| ------- | ------------: | ------------------: | -----------: |
| A       |          1-17 |                 200 |         3400 |
| B       |          7-48 |                 200 |         8400 |

### Problem C, D ve E

Problem C, D ve E için nokta sayısı 1 ile 10 arasında değişmektedir. Bu problemler için tüm değer aralıklarında yeterli sayıda veri üretilebildiği için ek filtreleme yapılmamıştır.

| Problem | Değer Aralığı | Aralık Başına Örnek | Toplam Örnek |
| ------- | ------------: | ------------------: | -----------: |
| C       |          1-10 |                 500 |         5000 |
| D       |           0-1 |                1000 |         2000 |
| E       |          1-48 |                 200 |         9600 |

## Eğitim ve Test Ayrımı

Veri setleri %80 eğitim ve %20 test olacak şekilde ayrılmıştır. Test için ayrı veri üretilmemiştir.

Ayrıca modeller, eğitim verisinin farklı oranları kullanılarak eğitilmiştir:

* %25 eğitim verisi
* %50 eğitim verisi
* %100 eğitim verisi

| Problem | Toplam Veri | Eğitim | Test | %25 Eğitim | %50 Eğitim | %100 Eğitim |
| ------- | ----------: | -----: | ---: | ---------: | ---------: | ----------: |
| A       |        3400 |   2720 |  680 |        680 |       1360 |        2720 |
| B       |        8400 |   6720 | 1680 |       1680 |       3360 |        6720 |
| C       |        5000 |   4000 | 1000 |       1000 |       2000 |        4000 |
| D       |        2000 |   1600 |  400 |        400 |        800 |        1600 |
| E       |        9600 |   7680 | 1920 |       1920 |       3840 |        7680 |

## Model Mimarisi

Giriş verisi iki boyutlu bir matris olduğu için CNN mimarisi tercih edilmiştir. Tüm problemler için aynı temel CNN mimarisi kullanılmış, yalnızca çıkış katmanı problem tipine göre değiştirilmiştir.

Model yapısı genel olarak şu şekildedir:

| Katman        | Açıklama                       |
| ------------- | ------------------------------ |
| Girdi         | 25x25 binary matris            |
| Conv2D        | İlk özellik çıkarım katmanı    |
| MaxPool2D     | Uzamsal boyut azaltma          |
| Conv2D        | İkinci özellik çıkarım katmanı |
| MaxPool2D     | Uzamsal boyut azaltma          |
| Flatten       | Düzleştirme                    |
| Linear + ReLU | Tam bağlantılı katman          |
| Linear        | Tahmin katmanı                 |

## Çıkış Katmanları ve Kayıp Fonksiyonları

| Problem | Çıkış Katmanı | Kayıp Fonksiyonu  |
| ------- | ------------- | ----------------- |
| A, B, E | 1 nöron       | MSELoss           |
| C       | 10 nöron      | CrossEntropyLoss  |
| D       | 1 nöron       | BCEWithLogitsLoss |

## Hiperparametre Optimizasyonu

Hiperparametre optimizasyonu için Grid Search yöntemi kullanılmıştır. Eğitim verisi kendi içinde %80 eğitim ve %20 doğrulama olarak ayrılmıştır.

Her kombinasyon belirli sayıda epoch boyunca eğitilmiş ve doğrulama setindeki başarıya göre en iyi hiperparametreler seçilmiştir.

### Arama Uzayı

| Parametre     | Değerler              |
| ------------- | --------------------- |
| Learning Rate | 0.001, 0.0005, 0.0001 |
| Batch Size    | 32, 64                |
| Conv Filters  | (32, 64), (64, 128)   |
| FC Size       | 128, 256              |

Toplamda her problem için 8 farklı konfigürasyon denenmiştir.

### Seçim Kriterleri

| Problem Tipi  | Seçim Kriteri      |
| ------------- | ------------------ |
| Regresyon     | En düşük MAE       |
| Sınıflandırma | En yüksek accuracy |

### Eğitim Ayarları

| Parametre               | Arama Aşaması | Eğitim Aşaması |
| ----------------------- | ------------: | -------------: |
| Epoch                   |            20 |            150 |
| Early Stopping Patience |            10 |             20 |
| Optimizer               |          Adam |           Adam |

Aşırı öğrenmeyi engellemek için Early Stopping kullanılmıştır. Eğitim verisinin %10’u doğrulama amacıyla ayrılmıştır.

### Seçilen En İyi Hiperparametreler

| Problem | Learning Rate | Batch Size | Conv Filters | FC Size |
| ------- | ------------: | ---------: | ------------ | ------: |
| A       |         0.001 |         32 | (64, 128)    |     256 |
| B       |        0.0001 |         32 | (32, 64)     |     256 |
| C       |         0.001 |         64 | (64, 128)    |     128 |
| D       |        0.0001 |         64 | (64, 128)    |     128 |
| E       |         0.001 |         32 | (32, 64)     |     128 |

## Deney Sonuçları

### Problem A - Minimum Manhattan Mesafesi

| Eğitim Oranı | Eğitim Veri Adeti | Epoch |   MAE |  RMSE |    R² |
| ------------ | ----------------: | ----: | ----: | ----: | ----: |
| %25          |               612 |    38 | 2.360 | 2.934 | 0.641 |
| %50          |              1224 |    54 | 1.752 | 2.197 | 0.799 |
| %100         |              2448 |    84 | 1.414 | 1.772 | 0.869 |

Model, en yakın Manhattan mesafesi problemini kabul edilebilir düzeyde öğrenebilmiştir. Eğitim verisi arttıkça tahmin performansı düzenli olarak iyileşmiştir.

Model özellikle 1-2 mesafe aralığında daha başarılı sonuçlar üretmiştir. Bunun nedeni, yakın nokta çiftlerinin CNN filtreleri tarafından daha kolay yakalanabilmesidir.

Daha büyük mesafelerde olası nokta yerleşimlerinin sayısı arttığı için varyans yükselmiş ve tahmin zorlaşmıştır.

### Problem B - Maksimum Manhattan Mesafesi

| Eğitim Oranı | Eğitim Veri Adeti | Epoch |   MAE |  RMSE |    R² |
| ------------ | ----------------: | ----: | ----: | ----: | ----: |
| %25          |              1512 |    75 | 2.431 | 3.124 | 0.934 |
| %50          |              3024 |    74 | 1.890 | 2.404 | 0.961 |
| %100         |              6048 |    60 | 1.648 | 2.118 | 0.969 |

Problem B’de model, Problem A’ya göre daha dengeli sonuçlar üretmiştir. Bunun nedeni maksimum Manhattan mesafesi için modelin genellikle en uçtaki noktaları tespit etmesinin yeterli olmasıdır.

Eğitim verisi arttıkça MAE değeri azalmış ve modelin performansı iyileşmiştir.

### Problem C - Nokta Sayısı Tahmini

| Eğitim Oranı | Eğitim Veri Adeti | Epoch | Accuracy |
| ------------ | ----------------: | ----: | -------: |
| %25          |               900 |    27 |    0.412 |
| %50          |              1800 |    41 |    0.792 |
| %100         |              3600 |    73 |    0.951 |

Problem C, en başarılı sonuç alınan problemdir. %100 eğitim verisi kullanıldığında model yaklaşık %95 doğruluğa ulaşmıştır.

Nokta sayısı düşük olduğunda model daha başarılıdır. Nokta sayısı arttıkça noktaların birbirine yaklaşması ve görüntüdeki karmaşıklığın artması nedeniyle sayım zorlaşmıştır.

Confusion Matrix incelendiğinde özellikle 7-9 aralığındaki sınıfların daha sık karıştırıldığı görülmüştür.

### Problem D - Tek / Çift Sınıflandırması

| Eğitim Oranı | Eğitim Veri Adeti | Epoch | Accuracy |
| ------------ | ----------------: | ----: | -------: |
| %25          |               360 |    32 |    0.567 |
| %50          |               720 |    22 |    0.493 |
| %100         |              1440 |    38 |    0.512 |

Model en kötü performansı Problem D’de göstermiştir. Doğruluk oranı yaklaşık %50 seviyesinde kalmış, yani model rastgele tahmine yakın sonuçlar üretmiştir.

Bu problemin başarısız olmasının temel nedeni, tek / çift ayrımının global bir sayma problemi olmasıdır. CNN mimarisi ise ağırlıklı olarak yerel özellikleri öğrenir. Yerel özellikler, nokta sayısının tek veya çift olduğu bilgisini doğrudan taşımaz.

Bu nedenle eğitim verisi artsa bile model bu problemi öğrenememiştir.

### Problem E - Köşe Mesafesi

| Eğitim Oranı | Eğitim Veri Adeti | Epoch |   MAE |  RMSE |    R² |
| ------------ | ----------------: | ----: | ----: | ----: | ----: |
| %25          |              1728 |    36 | 3.796 | 5.881 | 0.820 |
| %50          |              3456 |    59 | 3.523 | 5.786 | 0.826 |
| %100         |              6912 |    56 | 3.180 | 5.361 | 0.850 |

Problem E’de model, mesafe kavramını öğrenebilmesine rağmen tek / çift ayrımını doğru yapamadığı için sınırlı başarı göstermiştir.

Bu problem, Problem D’deki tek / çift kararını da içerdiğinden CNN mimarisinin global sayma konusundaki yetersizliği sonuçları olumsuz etkilemiştir.

Model bazı durumlarda mesafeyi yaklaşık olarak doğru tahmin etmiş ancak yanlış köşeye göre hesap yaptığı için büyük hatalar üretmiştir.

## Genel Sonuç Tablosu

| Problem                    | Tip           | Metrik   |   %25 |   %50 |  %100 |
| -------------------------- | ------------- | -------- | ----: | ----: | ----: |
| A - Min Manhattan Mesafesi | Regresyon     | MAE      | 2.360 | 1.752 | 1.414 |
| B - Max Manhattan Mesafesi | Regresyon     | MAE      | 2.431 | 1.890 | 1.648 |
| C - Nokta Sayısı           | Sınıflandırma | Accuracy | 0.412 | 0.792 | 0.951 |
| D - Tek / Çift             | Sınıflandırma | Accuracy | 0.567 | 0.493 | 0.512 |
| E - Köşe Mesafesi          | Regresyon     | MAE      | 3.796 | 3.523 | 3.180 |

## Değerlendirme

Elde edilen sonuçlar, CNN mimarisinin yerel örüntülere dayanan problemleri öğrenmede başarılı olduğunu göstermiştir. Problem A, B ve C’de model anlamlı sonuçlar üretmiştir.

Problem B, yalnızca en uçtaki noktaların tespitini gerektirdiği için Problem A’ya göre daha dengeli sonuçlar vermiştir.

Problem C’de nokta sayısı tahmini yüksek doğrulukla yapılabilmiştir. Ancak nokta sayısı arttıkça sınıflar arasındaki karışıklık da artmıştır.

Problem D, projenin en önemli negatif bulgusudur. Tek / çift belirleme problemi global bir sayma problemi olduğu için CNN mimarisi bu görevi öğrenememiştir.

Problem E’de ise model mesafe ilişkisini öğrenebilmesine rağmen doğru köşeyi seçebilmek için tek / çift bilgisine ihtiyaç duyduğundan düşük mesafe değerlerinde ciddi hatalar oluşmuştur.

Genel olarak eğitim verisinin artırılması, öğrenilebilir problemler olan A, B, C ve E’de performansı iyileştirmiştir. Ancak yapısal olarak CNN mimarisiyle öğrenilemeyen Problem D’de veri miktarının artması anlamlı bir iyileşme sağlamamıştır.

Bu sonuçlar, model başarısının yalnızca veri miktarına değil, problemin yapısına ve seçilen yapay sinir ağı mimarisine uygunluğuna da bağlı olduğunu göstermektedir.

## Kullanılan Teknolojiler

* C
* Python
* PyTorch
* CSV veri setleri

## Notlar

Bu çalışmada tüm problemler için aynı temel CNN mimarisi kullanılmıştır. Gelecek çalışmalarda özellikle global sayma gerektiren problemler için Transformer, RNN veya attention tabanlı mimarilerin denenmesi daha başarılı sonuçlar verebilir.
