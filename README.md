# Deri Kanseri Sınıflandırma Modeli (HAM10000)

Bu proje, HAM10000 veri setini kullanarak deri kanseri türlerini sınıflandırmak için bir derin öğrenme modeli (Convolutional Neural Network - CNN) geliştirmeyi amaçlamaktadır.

## Proje Tanımı

Bu çalışma, makine öğrenmesi ve derin öğrenme tekniklerini uygulayarak dermatologların deri lezyonlarını teşhis etmelerine yardımcı olabilecek bir sınıflandırma sistemi oluşturmayı hedeflemektedir. HAM10000 veri setindeki görüntüler kullanılarak model eğitilmiş ve çeşitli performans metrikleri ile değerlendirilmiştir.

## Veri Seti

Projede kullanılan veri seti, çeşitli deri lezyonlarının görüntülerini içeren **HAM10000 (Human Against Machine with 10000 Training Images)** veri setidir. Veri seti 7 farklı kategoriye ait görüntüleri içermektedir:

- akiec: Actinic keratoses and intraepithelial carcinomae
- bcc: Basal cell carcinoma
- bkl: Benign keratosis-like lesions
- df: Dermatofibroma
- nv: Melanocytic nevi
- vasc: Pyogenic granulomas and hemorrhage
- mel: Melanoma

Veri setine Kaggle üzerinden erişilebilir ve proje kapsamında indirilip kullanılmıştır.

## Metodoloji ve Modelin Çalışması

Projede genel olarak aşağıdaki adımlar izlenmiştir:

1.  **Veri Yükleme ve Hazırlık:**
    - HAM10000 metadata dosyası yüklenerek görüntülere ait bilgiler alınır.
    - Görüntüler belirtilen dizinlerden okunur.
    - Okunan görüntüler BGR formatından modelin daha iyi işleyebileceği RGB formatına dönüştürülür.
    - Tüm görüntüler standart bir boyuta (28x28 piksel) yeniden boyutlandırılır.
    - Görüntülerin piksel değerleri, modelin daha kararlı eğitilmesi için 0 ile 1 aralığına ölçeklenir (normalize edilir).
    - Görüntüler, modelin giriş katmanına uygun hale getirmek için düzleştirilir (flatten - tek boyutlu vektöre dönüştürülür).

    Veri setinden rastgele veya ilk birkaç örneği görselleştirmek, veriyi anlamak açısından önemlidir. Aşağıda veri setindeki ilk 5 görsel örneği görülebilir:

    ![İlk 5 Görsel Örneği]()

2.  **Veri Dengeleme:**
    - HAM10000 veri setindeki sınıflar arasında örnek sayısı açısından önemli dengesizlikler bulunmaktadır. Bu dengesizlik, modelin çoğunluk sınıflarına bias'lı olmasına neden olabilir.
    - Bu dengesizliği gidermek için `RandomOverSampler` veya `SMOTE` gibi yöntemler kullanılarak azınlık sınıflarına ait örnekler sentetik olarak çoğaltılır veya mevcut örnekler üzerinden yeniden örnekleme yapılır. Bu işlem eğitim veri setinin her sınıftan benzer sayıda örnek içermesini sağlar.

    Aşağıdaki grafik, veri dengeleme işlemi öncesi ve sonrası sınıf dağılımını göstermektedir:

    ![RandomOverSampler veya SMOTE Sonrası Sınıf Dağılımı]()

3.  **Model Mimarisi:**
    - Projede, basit ama etkili bir derin öğrenme modeli olan çok katmanlı bir Yapay Sinir Ağı (Dense Network) kullanılmıştır. Model Sequential API ile oluşturulmuştur.
    - Modelin giriş katmanı, düzleştirilmiş 28x28 piksellik görüntü verisini alır.
    - Ara katman olarak bir adet Yoğun (Dense) katman bulunur. Bu katman, giriş verisinden karmaşık örüntüleri öğrenmek için ReLU aktivasyon fonksiyonunu kullanır.
    - Aşırı öğrenmeyi azaltmak ve modelin genelleme yeteneğini artırmak için Dropout katmanı eklenmiştir. Bu katman, eğitim sırasında nöronların rastgele bir kısmını kapatarak modelin belirli özelliklere aşırı bağımlı olmasını engeller.
    - Çıkış katmanı, 7 farklı deri kanseri sınıfı için tahmin yapar ve her sınıf için bir olasılık değeri üretmek üzere Softmax aktivasyon fonksiyonunu kullanır.

4.  **Model Eğitimi:**
    - Model, sınıflandırma problemi için uygun olan 'sparse_categorical_crossentropy' kaybı fonksiyonu ve Adam optimize edici kullanılarak derlenir.
    - Modelin performansı eğitim ve doğrulama veri setleri üzerinde doğruluk (accuracy) metrikleri ile takip edilir.
    - Eğitim süreci, belirtilen epoch sayısı boyunca devam eder.
    - Modelin aşırı öğrenmesini önlemek ve en iyi performansı veren ağırlıkları otomatik olarak kaydetmek için Erken Durdurma (`EarlyStopping`) tekniği kullanılır. Doğrulama kaybı belirli bir epoch sayısı boyunca iyileşme göstermediğinde eğitim durdurulur ve en iyi ağırlıklar geri yüklenir.

    Aşağıdaki grafikler, eğitim süresince modelin eğitim ve doğrulama veri setleri üzerindeki kaybını ve doğruluğunu göstermektedir:

    ![Eğitim ve Doğrulama Grafikleri]()

5.  **Model Değerlendirmesi:**
    - Eğitilmiş modelin gerçek dünya performansı hakkında fikir edinmek için, eğitimde hiç görülmemiş ayrı bir test veri seti üzerinde değerlendirmesi yapılır.
    - Test veri seti üzerindeki genel kayıp ve doğruluk değerleri hesaplanır.
    - Modelin her bir sınıf üzerindeki performansını detaylı incelemek için Sınıflandırma Raporu (Classification Report) ve Karışıklık Matrisi (Confusion Matrix) oluşturulur.
    - Sınıflandırma raporu, Precision, Recall ve F1-Score gibi metriklerle her sınıf için ayrıntılı bilgi sunar.
    - Karışıklık matrisi, modelin hangi sınıfları doğru tahmin ettiğini ve hangi sınıfları birbiriyle karıştırdığını görsel olarak gösterir.

    Aşağıda modelin test veri seti üzerindeki Karışıklık Matrisi görülmektedir:

    ![Karışıklık Matrisi]()

    Ve modelin test veri seti üzerindeki Sınıflandırma Raporu çıktısı:

    ```
    (Sınıflandırma Raporu çıktısı buraya gelecek)
    ```

6.  **Örnek Tahminler:**
    - Modelin gerçek görüntüler üzerindeki tahminlerini göstermek için test setinden rastgele seçilen örnekler üzerinde tahminler yapılır.
    - Her bir örnek için orijinal görüntü, gerçek etiketi ve modelin tahmin ettiği etiket görselleştirilir. Bu, modelin ne kadar iyi çalıştığını veya hangi tür hatalar yaptığını anlamak için faydalıdır.

    Aşağıda test setinden seçilmiş örnek görüntüler ve modelin bunlara ait tahminleri gösterilmektedir:

    ![Örnek Tahmin Görselleri]()

## Kullanım

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  Bu GitHub deposunu yerel bilgisayarınıza klonlayın.
2.  Gerekli Python kütüphanelerini yükleyin. Bu kütüphaneler genellikle `tensorflow`, `keras`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` ve `imbalanced-learn`'dir. Bu kütüphaneleri içeren bir `requirements.txt` dosyası oluşturarak kurulumu kolaylaştırabilirsiniz:
    ```bash
    pip install -r requirements.txt
    ```
3.  Kaggle hesabınızdan API kimlik bilgilerini edinin ve ortamınıza kurun.
4.  Kaggle API'sini kullanarak HAM10000 veri setini indirin ve projenizin kök dizininde `ham10000` adında bir klasöre zip'ten çıkarın.
5.  Proje klasörünüzdeki Jupyter Notebook dosyasını (`ham10000.ipynb`) açın ve hücreleri sırasıyla çalıştırın.

## Dosyalar

- `ham10000.ipynb`: Projenin tüm kodunu içeren ana Jupyter Notebook dosyası.
- `my_model.keras`: Eğitilen derin öğrenme modelinin kaydedildiği dosya. (Notebook çalıştırıldığında oluşur)
- Veri seti indirildiğinde oluşacak klasörler (`ham10000`, `HAM10000_metadata.csv` vb.)

## Gereksinimler

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (`sklearn`)
- Imbalanced-learn (`imblearn`)
- Kaggle API (veri setini indirmek için)

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen pull request göndermeden önce değişikliklerinizi açıklayan bir issue açınız.

## Lisans

Bu proje MIT Lisansı altındadır. Detaylar için `LICENSE` dosyasına bakınız. (Eğer bir lisans dosyası ekleyecekseniz bu kısmı tutabilirsiniz.)

## İletişim

Sorularınız veya önerileriniz için lütfen iletişime geçin: [E-posta Adresiniz veya GitHub Profiliniz]
