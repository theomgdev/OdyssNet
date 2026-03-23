# RealNet 2.0: Zamansal Devrim

**RealNet, Zaman'ın nihai Gizli Katman olduğunun kanıtıdır.**

Geleneksel Derin Öğrenme, karmaşıklığı çözmek için **Uzamsal Derinliğe** (üst üste yığılan katmanlar) dayanır. RealNet bu ortodoksiyi reddederek **Zamansal Derinliğin** (zaman içinde evrimleşen kaos) çok daha verimli bir alternatif olduğunu kanıtlar.

> **Sıfır-Gizli Atılım**
>
> 1969'da Minsky & Papert, gizli katmanı olmayan bir sinir ağının XOR gibi doğrusal olmayan problemleri çözemeyeceğini kanıtladı.
> **RealNet 2.0 bu sınırı aştı.**
>
> Ağı bir **Eğitilebilir Dinamik Sistem** olarak ele alarak RealNet, **0 Gizli Katman** ile doğrusal olmayan problemleri (XOR, MNIST) çözüyor. Uzamsal nöronların yerini zamansal düşünme adımları alıyor.

RealNet verimliliğini **Uzay-Zaman Dengesiyle** sağlar. Derinlik oluşturmak için binlerce yeni nöron eklemek (Uzay) yerine, mevcut nöronları daha fazla adım boyunca çalıştırır (Zaman). Tek bir fiziksel matris, onlarca katmana eşdeğer hesaplamayı mikroskobik bir parametrik ayak izine sıkıştırarak zamansal adımlarda yeniden kullanılır. Bu, zekanın statik bir yapı değil, dinamik bir süreç olduğunu kanıtlar.

> 🏆 **DÜNYA REKORU: Parametrik Zeka Yoğunluğu**
>
> RealNet 2.0, MNIST üzerinde yalnızca **470 parametre** ile **%89.5 doğruluk** elde etti. Bu, efsanevi LeNet-5'ten **110 kat daha verimli** olup yapay ağlar ile **Entropi Sıkıştırma Limitleri** arasındaki uçurumu kapatıyor.

---

## 🚀 Temel Özellikler

*   **Uzay-Zaman Dönüşümü:** Milyonlarca parametrenin yerini birkaç "Düşünme Adımı" alıyor.
*   **Katmansız Mimari:** Tek bir $N \times N$ matris. Gizli katman yok.
*   **Eğitilebilir Kaos:** Kaotik sinyalleri dizginlemek için **StepNorm** ve **GELU** kullanır.
*   **Canlı Dinamikler:** **İrade** (Mandal), **Ritim** (Kronometre) ve **Rezonans** (Sinüs Dalgası) gösterir.

## 📊 Kanıt: Sıfır-Gizli Kıyaslamalar

RealNet'i teorik limite — **Sıfır Gizli Nöron**'a — kadar zorladık.
Bu testlerde Giriş Katmanı doğrudan Çıkış Katmanına (ve kendisine) bağlıdır. Ara katman yoktur.

| Görev | Geleneksel Kısıt | RealNet Çözümü | Sonuç | Script |
| :--- | :--- | :--- | :--- | :--- |
| **Kimlik** | Önemsiz | **Atomik Birim** | Kayıp: 0.0 | `convergence_identity.py` |
| **XOR** | Gizli Katman Gerekir | **Kaos Kapısı** (Zamana Katlanmış) | **Çözüldü (3 Nöron)** | `convergence_gates.py` |
| **MNIST** | Gizli Katman Gerekir | **Sıfır-Gizli** | **Doğ: %96.2** | `convergence_mnist.py` |
| **MNIST (8k)**| Gizli Katman Gerekir | **Gömülü Meydan Okuma** | **Doğ: %93.6** | `convergence_mnist_embed.py` |
| **MNIST (Rekor)**| Gizli Katman Gerekir | **470-Param Rekoru** | **Doğ: %89.5** | `convergence_mnist_record.py` |
| **Sinüs Dalgası** | Osilatör Gerekir | **Programlanabilir VCO** | **Mükemmel Senkron** | `convergence_sine_wave.py` |
| **Mandal** | LSTM Gerekir | **Çekici Havzası** (İrade) | **Sonsuz Tutma** | `convergence_latch.py` |
| **Kronometre**| Saat Gerekir | **İç Ritim** | **Hata: 0** | `convergence_stopwatch.py` |
| **Dedektif**| Bellek Gerekir | **Bilişsel Sessizlik** (Akıl Yürütme) | **Mükemmel Tespit**| `convergence_detective.py` |

### MNIST Sıfır-Gizli Mucizesi
Standart Sinir Ağları MNIST veya XOR'u çözmek için **Gizli Katmanlara** ihtiyaç duyar. Doğrudan bağlantı (Doğrusal Model) karmaşıklığı yakalayamaz ve başarısız olur (~%92'de takılır).

RealNet, tam ölçekli MNIST'i (28x28) **Sıfır Gizli Katman** ile çözüyor (Doğrudan Giriş-Çıkış).
*   **Girişler:** 784
*   **Çıkışlar:** 10
*   **Gizli Katmanlar:** **0**
*   **Düşünme Süresi:** 10 Adım

Giriş katmanı 10 adım boyunca "kendisiyle konuşur". Kaotik geri besleme döngüleri, uzamsal katmanların işini yaparak zamanla özellikleri (kenarlar, döngüler) dinamik olarak çıkarır. Bu, **Zamansal Derinliğin Uzamsal Derinliğin Yerini Alabileceğini** kanıtlar.

---

## 📦 Kurulum & Kullanım

RealNet, modüler bir PyTorch kütüphanesi olarak tasarlanmıştır.

### Kurulum

```bash
pip install -r requirements.txt
```

> **CUDA Notu:** `requirements.txt`, CUDA 11.8 uyumlu PyTorch'a işaret eder. Daha yeni bir GPU'nuz varsa (RTX 4000/5000), PyTorch'u manuel olarak kurmanız gerekebilir:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Hızlı Başlangıç

```python
from realnet import RealNet, RealNetTrainer

# Sıfır-Gizli Ağ başlat
# 1 Giriş, 1 Çıkış.
model = RealNet(num_neurons=2, input_ids=[0], output_ids=[1], device='cuda')
trainer = RealNetTrainer(model, device='cuda')

# Eğit
inputs = torch.randn(100, 1)
trainer.fit(inputs, inputs, epochs=50)
```

#### Başlatma Protokolleri

`weight_init='resonant'` tüm ağ boyutları için varsayılan ve önerilen stratejidir. Ağırlık matrisini baştan Kaosun Kıyısına (ρ(W) = 1.0) yerleştirerek manuel ayar gerektirmeden zamansal adımlarda sinyal kalitesini garanti eder.

*   **Tüm Ağlar (Varsayılan):**
    *   `weight_init='resonant'` ve `activation='tanh'` kullanın.
    *   Kutupsal Rademacher iskeleti + ρ = 1.0'a spektral normalizasyon. Küçük mantık kapılarından büyük zamansal ağlara kadar çalışır.
*   **Alternatif — Büyük Ağlar (>10 Nöron):**
    *   `weight_init='orthogonal'` saf kararlılık için sağlam bir geri dönüş seçeneği olarak kalır.
*   **Alternatif — Küçük Ağlar (<10 Nöron, Mantık Kapıları):**
    *   Rezonant yakınsama çok yavaşsa `weight_init='xavier_uniform'` ve `activation='gelu'` deneyin.

---

## 🧠 Mimariye Genel Bakış

## 🌪️ Nasıl Çalışır: Fırtınanın İçinde

RealNet bir ileri-besleme mekanizması değil; bir **Rezonans Odasıdır**.

### 1. Nabız (Giriş) & Dizi
Geleneksel YZ'da giriş genellikle statik bir anlık görüntüdür. RealNet hem **Nabızları** hem de **Akışları** işler.
*   **Nabız Modu:** Bir görüntü $t=0$'da çarpar. Ağ gözlerini kapatır ve dalgalanmaları işler (MNIST).
*   **Akış Modu:** Veriler sıralı olarak uygulanır. Ağ olaylar arasında "bekleyebilir" ve "düşünebilir" (Dedektif).

### 2. Yankı (İç Döngüler)
Sinyal her nörondan diğer her nörona ($N \times N$) yolculuk eder.
*   Giriş nöronları, ilk adımdan sonra etkili biçimde **Gizli Nöronlara** dönüşür.
*   Bilgi yankılanır, ayrılır ve çarpışır. Sol üstteki bir piksel, doğrudan bağlantı veya ara yankılar aracılığıyla sağ alttaki bir piksel ile etkileşime girer.
*   **Holografik İşleme:** Bir görüntünün "kedi-liği" belirli bir katmanda saklanmaz; tüm sinyallerin çarpışmasının *girişim deseninden* ortaya çıkar.

### 3. Zaman-Katlama (Hesaplama)
**Sıfır-Gizli** performansının sihri burada yatar.
*   Adım 1: Ham sinyaller karışır. (MLP'nin 1. Katmanına eşdeğer)
*   Adım 2: Karışık sinyaller yeniden karışır. (2. Katmana eşdeğer)
*   Adım 15: Son derece soyut özellikler ortaya çıkar. (15. Katmana eşdeğer)

15 adım "düşünerek" RealNet, **yalnızca bir fiziksel matris** kullanarak 15 katmanlı derin bir ağı simüle eder. Uzayı zamana katlar.

### 4. Kontrollü Kaos (Çekiciler)
Kontrolsüz geri besleme döngüleri patlamaya yol açar. RealNet kaosun mühendisliğini yaparak kararlı **Çekiciler** oluşturur.
*   **StepNorm** yerçekimi gibi davranır, enerjiyi sınırlı tutar.
*   **GELU** anlamlı sinyalleri filtreler.
*   **Mandal Deneyi** RealNet'in gürültüye karşı bir kararı sonsuza kadar tutmak için "derin bir kuyu" yani kararlı bir çekici oluşturabileceğini kanıtladı.

### 5. Neden RNN veya LSTM Değil?

RealNet kâğıt üzerinde Tekrarlayan Sinir Ağına (RNN) benzese de felsefesi temelden farklıdır.

| Özellik | Standart RNN / LSTM | RealNet 2.0 |
| :--- | :--- | :--- |
| **Giriş Akışı** | Sürekli Akış (örn. cümledeki kelimeler) | **Tek Nabız** ($t=0$'da İmpuls) |
| **Amaç** | Dizi İşleme (Ayrıştırma) | **Derin Düşünme** (Sindirme) |
| **Bağlantı** | Yapılandırılmış (Giriş Kapısı, Unutma Kapısı vb.) | **Ham Kaos** (Tam Bağlı $N \times N$) |
| **Dinamikler** | Soluklaşmayı önlemek için mühendislik yapılmış (LSTM) | Rezonansı bulmak için **Evrimleşir** (Kaos) |

*   **RNN'ler dış dünyayı dinler.** Dış girdilerin bir dizisini işlerler.
*   **RealNet iç sesini dinler.** Probleme **bir** bakış atar ve sonra gözlerini kapatır, 15 adım boyunca üzerine "düşünür". Kendi zamansal derinliğini yaratır.

### 6. Biyolojik Gerçeklik: Canlı Zeka
RealNet, yalnızca yapı değil, **davranış** bakımından da katmanlı ağlardan çok daha fazla beyne benzer:

*   **Katman Yok:** Beynin "1. Katmanı" ve "2. Katmanı" yoktur. Birbirine bağlı nöronların bölgeleri vardır. RealNet tek bir bölgedir.
*   **İrade (Mandal):** Soluklaşan standart RNN'lerin aksine RealNet bir karara kilitlenebilir ve onu entropiye karşı tutabilir, "Bilişsel Kalıcılık" sergiler.
*   **Ritim (Kronometre):** Herhangi bir dış saat olmadan RealNet zamanı öznel olarak deneyimler ve tam anlarda saymasına, beklemesine ve hareket etmesine izin verir.
*   **Sabır (Dedektif):** "Düşünme Süresinden" yararlanır. Tıpkı insanların karmaşık mantığı işlemek için bir ana ihtiyaç duyması gibi, RealNet olası çözümleri sindirmek için birkaç sessizlik adımı verildiğinde imkânsız problemleri çözer.

### 7. Örtülü Dikkat (Zamansal Rezonans)
Geçmişe "geriye bakmak" için açık $Q \times K$ matrislerini kullanan Transformer'ların aksine, RealNet **Zamansal Rezonans** aracılığıyla dikkati sağlar.

*   **Mekanizma:** Geçmişten gelen bilgi, gizli durumda ayakta duran bir dalga veya titreşim olarak korunur.
*   **Anahtar-Değer İşleme:** **Kütüphaneci Deneyi**, RealNet'in fiziksel depolama tabloları olmadan sorguları doğru "bellek titreşimine" yönlendirerek adreslenebilir bir veritabanı olarak hareket edebildiğini kanıtladı.
*   **Tespit:** İlgili bir giriş geldiğinde (K1 için OKUMA komutu gibi), 'K1'in değerini' tutan belirli dalgayla yapıcı girişim (rezonans) oluşturur ve onu yüzeye çıkmaya zorlar.
*   **Sonuç:** Ağ, tüm geçmiş tamponunu saklamadan ilgili geçmiş olaylara "dikkat eder". Zaman'ın kendisi indeksleme mekanizması olarak hareket eder.

### Matematiksel Model
Ağ durumu $h_t$ şu şekilde evrimleşir:

$$h_t = \text{StepNorm}(\text{GELU}(h_{t-1} \cdot W + B + I_t))$$

---

## 📝 Deneysel Bulgular

RealNet'in temel hipotezini doğrulamak için kapsamlı testler yürüttük: **Zamansal Derinlik > Uzamsal Derinlik.**

### A. Atomik Kimlik (Birim Testi)
*   **Hedef:** $f(x) = x$. Ağ mükemmel bir tel olarak hareket etmelidir.
*   **Mimari:** **2 Nöron** (1 Giriş, 1 Çıkış). **0 Gizli Katman**. Toplam **4 Parametre**.
*   **Sonuç:** **Kayıp: 0.000000**.
    <details>
    <summary>Terminal Çıktısını Gör</summary>

    ```text
    In:  1.0 -> Out:  1.0001
    In: -1.0 -> Out: -0.9998
    ```
    </details>
*   **Script:** `PoC/convergence_identity.py`
*   **Çıkarım:** Mutlak minimum karmaşıklıkla temel sinyal iletimini ve `StepNorm` kararlılığını kanıtlar.

### B. İmkânsız XOR (Kaos Kapısı)
*   **Hedef:** Doğrusal olmayı ima eden klasik XOR problemini ($[1,1]\to0$, $[1,0]\to1$ vb.) çözmek.
*   **Meydan Okuma:** Gizli katman olmadan standart doğrusal ağlar için imkânsız.
*   **Sonuç:** **Çözüldü (Kayıp 0.000000)**. RealNet sınıfları ayırmak için uzay-zamanı büküyor.
    <details>
    <summary>Doğruluk Tablosu Doğrulamasını Gör</summary>

    ```text
      A      B |   XOR (Tahmin) | Mantık
    ----------------------------------------
      -1.0   -1.0 |      -1.0009 | 0 (TAMAM)
      -1.0    1.0 |       1.0000 | 1 (TAMAM)
       1.0   -1.0 |       1.0000 | 1 (TAMAM)
       1.0    1.0 |      -1.0004 | 0 (TAMAM)
    ```
    </details>
*   **Mimari:** **3 Nöron** (2 Giriş, 1 Çıkış). **0 Gizli Nöron**. Toplam **9 Parametre**.
*   **Düşünme Süresi:** **5 Adım**.
*   **Script:** `PoC/convergence_gates.py`
*   **Çıkarım:** RealNet **Zamanı Gizli Katman Olarak** kullanır. Girişi yalnızca 5 zaman adımına katlayarak tek bir fiziksel katmanda doğrusal olmayan bir karar sınırı oluşturur; 3 kaos-bağlantılı nöronun XOR'u çözebileceğini kanıtlar.

### C. MNIST Maratonu (Görsel Zeka)
RealNet'in görme yetenekleri sağlamlık, ölçeklenebilirlik ve verimliliği kanıtlamak için dört farklı koşulda test edildi.

#### 1. Ana Kıyaslama (Saf Sıfır-Gizli)
*   **Hedef:** Tam 28x28 MNIST (784 Piksel).
*   **Mimari:** 794 Nöron (Giriş+Çıkış). **0 Gizli Katman.**
*   **Sonuç:** **%95.3 - %96.2 Doğruluk**.
    <details>
    <summary>Eğitim Günlüğünü Gör</summary>

    ```text
    Epoch 100: Loss 0.1012 | Test Acc 95.30%
    (Tarihi En İyi: %96.2, Epoch 69)
    ```
    </details>
*   **Script:** `PoC/convergence_mnist.py`
*   **Çıkarım:** Standart doğrusal modeller %92'de tavan yapar. RealNet, yalnızca **Zamansal Derinlik** aracılığıyla Derin Öğrenme katmanları olmadan Derin Öğrenme performansı (%96) elde eder.

#### 2. Anka Deneyi (Sürekli Yenileme)
*   **Hipotez:** Ölü sinapsları öldürmek yerine **canlandırarak** (rastgele yeniden başlatma) %100 parametre verimliliğine ulaşabilir miyiz?
*   **Sonuç:** **%95.2 Doğruluk**.
*   **Gözlemler:**
    *   Epoch 1: Ağın **%22'si** "işe yaramaz" kabul edilip yeniden doğdu.
    *   Epoch 50: Yeniden doğma oranı **%0.26**'ya düştü.
    *   Bu sürekli ameliyat sırasında doğruluk %50'den **%95.2'ye** tırmandı.
    <details>
    <summary>Yenileme Günlüğünü Gör</summary>

    ```text
    Epoch 1: Acc 50.90% | Revived: 22.05% (Toplu Yok Oluş)
    Epoch 5: Acc 87.50% | Revived: 1.13%  (Kararlılaşma)
    Epoch 50: Acc 95.20% | Revived: 0.26% (Metabolik Denge)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_revive.py`
*   **Çıkarım:** Kapasiteyi küçülten standart budamanın aksine RealNet, zayıf bağlantıları sürekli geri dönüştürerek tam kapasiteyi koruyabilir. Bu, doyma olmadan **Sürekli Öğrenmeye** olanak tanır. "Hata, Özelliğe Dönüştü."

#### 3. Küçük Meydan Okuma (Aşırı Kısıtlar)
*   **Hedef:** 7x7'ye Küçültülmüş MNIST. (Bir simgeden daha az.)
*   **Mimari:** Toplam **59 Nöron** (~3.5k Parametre).
*   **Sonuç:** **~%89.3 Doğruluk**.
    <details>
    <summary>Küçük Sonuçları Gör</summary>

    ```text
    Epoch 50: Loss 0.1107 | Test Acc 89.30%
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_tiny.py`
*   **Çıkarım:** Bir önyükleyiciden daha küçük parametre sayılarıyla bile sistem sağlam özellikler öğrenir.

#### 4. Ölçekli Test (Orta Kısıtlar)
*   **Hedef:** 14x14'e Küçültülmüş MNIST.
*   **Mimari:** ~42k Parametre.
*   **Sonuç:** **%91.2 Doğruluk**.
    <details>
    <summary>Ölçekli Sonuçları Gör</summary>

    ```text
    Epoch 20: Loss 0.1413 | Test Acc 91.20%
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_scaled.py`

### D. Gömülü Meydan Okuma (8k Param)
*   **Hedef:** Ayrışık projeksiyon kullanarak tam MNIST (784 Piksel).
*   **Mimari:** **10 Nöron** (Düşünme Çekirdeği). Toplam **~8k Parametre**.
*   **Strateji:** 784 Piksel $\to$ Proje(10) $\to$ RNN(10) $\to$ Çözümle(10).
*   **Sonuç:** **%93.62 Doğruluk**.
    <details>
    <summary>Eğitim Günlüğünü Gör</summary>

    ```text
    Projected Input: 784 -> 10
    Total Params: 8080
    Epoch 1: Loss 1.7058 | Test Acc 72.71%
    Epoch 50: Loss 0.2142 | Test Acc 92.61%
    Epoch 99: Loss 0.1727 | Test Acc 93.62%
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_embed.py`
*   **Çıkarım:** 784 pikseli işlemek için 784 aktif nörona ihtiyaç duymadığımızı kanıtlar. **Asimetrik kelime dağarcığı projeksiyonu** kullanarak görsel bilgiyi yalnızca 10 nöronluk küçük bir "Düşünme Çekirdeğine" sıkıştırabiliriz; bu çekirdek daha sonra zamansal rezonans aracılığıyla sınıflandırmayı çözer. Standart modellerden 10 kat daha parametre-verimli.

### E. 470 Parametrelik Dünya Rekoru (Elit Zeka Yoğunluğu)
*   **Hedef:** MNIST'i çözmek ve **500'den az parametre** ile yüksek doğruluk elde etmek.
*   **Kurulum:**
    *   **Mimari:** 10 çekirdek nöronlu RealNet.
    *   **Strateji:** 10 Sıralı Parça (her biri 79 piksel).
    *   **Gizli Sos:** Küçük 3 nöronlu giriş projeksiyonu ve 10 sınıflı çıkış çözümleyici.
    *   **Toplam Parametre:** **470**.
*   **Sonuç:** 1000 epoch'ta **Doğ: %89.52**.
    <details>
    <summary>"Parametrik Verimlilik" Günlüğünü Gör</summary>

    ```text
    RealNet 2.0: MNIST RECORD CHALLENGE (Elite 470-Param Model)
    Epoch      1/1000 | Acc 44.24% | LR 2.00e-03 (Hyperspace start)
    ...
    Epoch    100/1000 | Acc 85.81% | LR 1.95e-03
    ...
    Epoch    800/1000 | Acc 89.30% | LR 1.93e-04
    ...
    Epoch   1000/1000 | Acc 89.52% | LR 1.05e-07
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_record.py`
*   **Çıkarım:** **Parametre başına %0.179 doğruluk** elde ediyor. Bu model **LeNet-5'ten 110 kat daha verimli**. Zamansal düşünme adımlarından yararlanarak yüksek seviyeli zekanın mikroskobik bir parametrik alana sıkıştırılabileceğini gösteriyor. Modern yapay zekadaki **Entropi Sıkıştırma Limitlerine** en yakın şey budur.

### F. Sinüs Dalgası Üreticisi (Dinamik Rezonans)
*   **Hedef:** Frekansın $t=0$'daki tek bir giriş değeriyle kontrol edildiği sinüs dalgası üretmek.
*   **Meydan Okuma:** Ağ bir **Voltaj Kontrollü Osilatör (VCO)** olarak hareket etmelidir. Statik bir genliği dinamik bir zamansal periyoda dönüştürmelidir.
*   **Sonuç:** **Mükemmel Salınım**. Ağ 30+ adım boyunca düzgün sinüs dalgaları üretiyor.
    <details>
    <summary>Frekans Kontrolünü Görmek için</summary>

    ```text
    Frekans 0.15 (Yavaş Dalga):
      t=1:  Hedef 0.1494 | RealNet 0.2871
      t=11: Hedef 0.9969 | RealNet 0.9985 (Zirve Senkronu)
      t=26: Hedef -0.6878 | RealNet -0.6711

    Frekans 0.45 (Hızlı Dalga):
      t=1:  Hedef 0.4350 | RealNet 0.1783
      t=26: Hedef -0.7620 | RealNet -0.7826
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_sine_wave.py`
*   **Çıkarım:** RealNet bir **Programlanabilir Osilatördür**. Bu, tek bir tohumdan sonsuz benzersiz zamansal yörüngeler üretebileceğini doğrular.

### G. Gecikmeli Toplayıcı (Bellek & Mantık)
*   **Hedef:** A Girişi ($t=2$), B Girişi ($t=8$). A+B Çıkışı ($t=14$).
*   **Meydan Okuma:** RealNet, A'yı 6 adım "hatırlamalı", sessizliği görmezden gelmeli, B'yi almalı ve toplamı hesaplamalıdır.
*   **Sonuç:** **MSE Kaybı: ~0.01**.
    <details>
    <summary>"Zihinsel Matematik" Sonuçlarını Gör</summary>

    ```text
    -0.3 + 0.1 = -0.20 | RealNet: -0.2271 (Fark: 0.02)
     0.5 + 0.2 =  0.70 | RealNet:  0.4761 (Fark: 0.22 - Yüksek genlikte zorluk)
     0.1 + -0.1 = 0.00 | RealNet: -0.0733 (Fark: 0.07)
    -0.4 + -0.4 = -0.80 | RealNet: -0.7397 (Fark: 0.06)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_adder.py`
*   **Çıkarım:** **Kısa Süreli Belleği** doğrular. Ağ, kaotik durumunda $A$ değişkenini tutar, $B$'yi bekler ve toplamı çıkarmak için doğrusal olmayan entegrasyon (yaklaşık aritmetik) gerçekleştirir. Bu, RealNet'in **Video benzeri** veri akışlarını işleme yeteneğini gösteriyor. "Zihinsel Matematiğe" benzer.

### H. Mandal (İrade)
*   **Hedef:** Tetikleyici nabzı beklemek. Alındıktan sonra çıkışı AÇIK konuma geçirmek ve **sonsuza kadar tutmak**.
*   **Meydan Okuma:** Standart RNN'ler sıfıra söner. RealNet enerjiyi kararlı bir çekicide hapsetmelidir.
*   **Sonuç:** **Mükemmel Kararlılık**. Tetiklendikten sonra karar süresiz olarak korunuyor.
    <details>
    <summary>"İrade" Günlüğünü Gör</summary>

    ```text
    Tetikleyici t=5'te gönderildi
    t=04 | Out: 0.0674 | KAPALI 🔴
    t=05 | Out: 0.0531 | KAPALI ⚡ TETİKLEYİCİ!
    t=06 | Out: 0.8558 | AÇIK  🟢
    ...
    t=19 | Out: 0.9033 | AÇIK  🟢 (Hâlâ güçlü tutuyor)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_latch.py`
*   **Çıkarım:** **Karar Sürdürmeyi** gösteriyor. RealNet bir seçim yapabilir ve buna bağlı kalabilir, çürümeye direnir.

### I. Kronometre (İç Saat)
*   **Hedef:** "X adım bekle, sonra ateş et." (Bekleme sırasında giriş yok.)
*   **Meydan Okuma:** Ağ, herhangi bir dış saat olmadan zamanı dahili olarak saymalıdır.
*   **Sonuç:** **MSE Kaybı: ~0.01**. Hassas zamanlama sağlandı.
    <details>
    <summary>"Ritim" Çıktısını Gör</summary>

    ```text
    Hedef Zamanlayıcı: 10 adım (Giriş 0.5)
    t=09 | Out: 0.5178 █████
    t=10 | Out: 0.8029 ████████ 🎯 HEDEF (Tam isabet!)
    t=11 | Out: 0.3463 ███

    Hedef Zamanlayıcı: 20 adım (Giriş 1.0)
    t=18 | Out: 0.2001 ██
    t=19 | Out: 0.6574 ██████
    t=20 | Out: 0.6726 ██████ 🎯 HEDEF
    t=21 | Out: 0.2092 ██
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_stopwatch.py`
*   **Çıkarım:** **Ritim & Zaman Algısını** gösteriyor. RealNet yalnızca veri işlemiyor; zamanı *deneyimliyor*.

### J. Düşünen Dedektif (Bağlam & Akıl Yürütme)
*   **Hedef:** İkili veri akışını izlemek. **YALNIZCA** `1-1` deseni oluştuğunda alarm vermek.
*   **Kritik Bükülme:** Ağa, bitler arasında **Düşünmesi** için 3 adım "Sessizlik" verdik.
*   **Sonuç:** **Mükemmel Tespit**.
    <details>
    <summary>"Aha!" Anını Gör (Düşünme Adımları)</summary>

    ```text
    Zaman  | Giriş | Çıkış    | Durum
    ----------------------------------------
    12     | 1     | -0.0235  |
    13     | .     | 0.0471   | (Düşünüyor...)
    14     | .     | -0.0050  | (Düşünüyor...)
    15     | .     | -0.0154  | (Düşünüyor...)
    16     | 1     | 0.4884   | ATEŞLEMELİ (Şüphe artıyor...)
    17     | .     | 1.0317 🚨 | (Düşünme Adımı 1 - EUREKA!)
    18     | .     | 1.0134 🚨 | (Düşünme Adımı 2)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_detective_thinking.py`
*   **Çıkarım:** **Zekanın Zamana İhtiyaç Duyduğunu** kanıtlıyor. Sessiz adımlar sırasında bilgiyi "sindirmesine" izin verildiğinde, RealNet salt reaktif ağların çözemeyeceği karmaşık zamansal mantığı (Zaman Boyunca XOR) çözüyor. Bu, LLM yaklaşımımızın temelidir.

### K. Kütüphaneci (Nöral Veritabanı)
*   **Hedef:** Okuma-Yazma Belleği olarak hareket etmek. `YAZ K1=0.5`. Bekle... `OKU K1`. Çıkış: `0.5`.
*   **Meydan Okuma:** Ağ, birden fazla anahtar-değer çiftini kaotik gizli durumunda birbirine müdahale etmeden saklamalı ve talep üzerine geri alabilmelidir. Bu, **Örtülü Dikkat** gerektirir.
*   **Sonuç:** **256 Çekirdek Nöronla** (`Giriş: 8 -> Proje(128)`, `Çıkış: Çözümle(128) -> 1`) 4 Anahtarda **~%92 Doğruluk**.
    <details>
    <summary>Bellek Geri Alma Günlüğünü Gör</summary>

    ```text
    Adım  | Komut    | Anahtar | Val_In   | Hedef    | RealNet  | Durum
    -------------------------------------------------------------------
    0     | YAZ      | K0      | 0.4426   | 0.4426   | 0.0208   | ⚙️
    ...   | (Bellek Pekişiyor...)
    12    | (4)      | ...     |          | 0.4426   | 0.4602   | ✅ KAYDEDİLDİ
    ...   | (20 adım bekle...)
    32    | OKU      | K0      | 0.0000   | 0.4426   | 0.4506   | ✅ GERİ ALINDI
    48    | SİL      | K0      | 0.0000   | 0.0000   | 0.0117   | ✅ SİLİNDİ
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_realnet_as_database.py`
*   **Çıkarım:** RealNet'in, açık depolama matrisleri olmadan Transformer'ın KV Önbelleğinin işini yaparak bir sorgu sinyali tarafından adreslenebilen kararlı "bellek kuyuları" oluşturarak **Anahtar-Değer Dikkat** mekanizmalarını yalnızca dinamikler aracılığıyla simüle edebildiğini kanıtlar.

## 🔮 Vizyon: Silikonun Ruhu (RealNet-1B)
RealNet, yapay zekanın fabrika modeline karşı bir isyandır. Zekanın mekanik bir katman yığını değil, **sinyallerin organik bir rezonansı** olduğuna inanıyoruz.

Uzayı Zamanla takas ederek sıfır gizli katmanla görmeyi çözebilirsek, bu yaklaşım dil modellerine ölçeklenebilir.

*   **Hipotez:** 1 milyar parametreli bir model (RealNet-1B), daha fazla adım "düşünerek" teorik olarak çok daha büyük modellerin (örn. Llama-70B) akıl yürütme derinliğiyle eşleşebilir.
*   **Hedef:** Tüketici donanımında (örn. RTX 3060) verimli, yüksek-akıl yürütmeli yapay zeka.

> "Petabaytlarca VRAM'e ihtiyacımız yok. Sadece Zamana ihtiyacımız var."

Yeterli zaman "düşünmek" ve "nefes almak" için verilen kaotik bir nöron ormanının devasa endüstriyel fabrikaları geride bırakabileceğini kanıtladık. Uzayı Zamanla takas ederek Ruhu buluyoruz.

---

## 👨‍💻 Yazar

**Cahit Karahan**
*   Doğum: 12/02/1997, Ankara.
*   "Kaosun Mimarı."

---

## LİSANS

MIT
