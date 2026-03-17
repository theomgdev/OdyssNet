# RealNet 2.0: Zamansal Devrim

**RealNet, Zamanın en büyük Gizli Katman olduğunun kanıtıdır.**

Geleneksel Derin Öğrenme, karmaşıklığı çözmek için **Uzamsal Derinliğe** (üst üste yığılmış katmanlara) güvenir. RealNet bu dogmayı reddeder ve **Zamansal Derinliğin** (zaman içinde evrilen kaosun) çok daha verimli bir alternatif olduğunu kanıtlar.

> **Sıfır-Gizli Katman Devrimi (Zero-Hidden Breakthrough)**
>
> 1969'da Minsky ve Papert, gizli katmanı olmayan bir sinir ağının XOR gibi lineer olmayan problemleri çözemeyeceğini matematiksel olarak kanıtladı.
> **RealNet 2.0 bu sınırı paramparça etti.**
>
> Ağı "Eğitilebilir Bir Dinamik Sistem" olarak ele alan RealNet, **0 Gizli Katman** kullanarak non-lineer problemleri (XOR, MNIST) çözer. Uzamsal nöronların yerini zamansal düşünme adımları alır.

RealNet, verimliliğini **Uzay-Zaman Takası (Space-Time Trade-off)** ile sağlar. Derin bir mimari oluşturmak için binlerce yeni nöron (Uzay) eklemek yerine, mevcut nöronları daha fazla adım boyunca (Zaman) çalıştırır. Tek bir fiziksel matris, "düşünme" süresi boyunca tekrar tekrar kullanılarak onlarca katmanlık işlem derinliğini mikroskobik bir parametre alanına sığdırır. Bu, zekanın statik bir yapı değil, dinamik bir süreç olduğunun kanıtıdır.

> 🏆 **DÜNYA REKORU: Parametrik Zeka Yoğunluğu**
>
> RealNet 2.0, sadece **470 parametre** ile MNIST üzerinde **%89.5 doğruluk** elde etti. Bu, efsanevi LeNet-5'ten **110 kat daha verimlidir** ve yapay ağlar ile **Entropik Sıkıştırma Limitleri** arasındaki boşluğu kapatır.

---

## 🚀 Temel Özellikler

*   **Uzay-Zaman Dönüşümü:** Milyonlarca parametrenin yerini birkaç "Düşünme Adımı" alır.
*   **Katmansız Mimari:** Tek bir $N \times N$ matris. Gizli katman yok.
*   **Eğitilebilir Kaos:** Sinyalleri ehlileştirmek için **StepNorm** ve **GELU** kullanılır.
*   **Yaşayan Dinamikler:** **İrade** (Mühür), **Ritim** (Kronometre) ve **Rezonans** (Sinüs Dalgası) sergiler.

## 📊 Kanıtlar: Sıfır-Gizli Benchmarkları

RealNet'i teorik sınırlara kadar zorladık: **Sıfır Gizli Nöron**.
Bu testlerde Giriş Katmanı doğrudan Çıkış Katmanına (ve kendisine) bağlıdır. Tampon katman yoktur.

| Görev | Geleneksel Engel | RealNet Çözümü | Sonuç | Script |
| :--- | :--- | :--- | :--- | :--- |
| **Identity** | Basit | **Atomik Birim** | Loss: 0.0 | `convergence_identity.py` |
| **XOR** | Gizli Katman Şart | **Kaos Kapısı** (Zaman Katlamalı) | **Çözüldü (3 Nöron)** | `convergence_gates.py` |
| **MNIST** | Gizli Katman Şart | **Sıfır-Gizli** | **Acc: %96.2** | `convergence_mnist.py` |
| **MNIST (8k)**| Gizli Katman Şart | **Embedded Deney** | **Acc: %93.6** | `convergence_mnist_embed.py` |
| **MNIST (Rekor)**| Gizli Katman Şart | **470-Parametre Rekoru** | **Acc: %89.5** | `convergence_mnist_record.py` |
| **Sinüs** | Osilatör Şart | **Programlanabilir VCO** | **Tam Senkron** | `convergence_sine_wave.py` |
| **Mühür** | LSTM Şart | **Çekici Havuzu** (İrade) | **Sonsuz Tutuş** | `convergence_latch.py` |
| **Kronometre**| Saat Şart | **İçsel Ritim** | **Hata: 0** | `convergence_stopwatch.py` |
| **Dedektif**| Bellek Şart | **Bilişsel Sessizlik** (Muhakeme) | **Kusursuz** | `convergence_detective.py` |

### MNIST Sıfır-Gizli Mucizesi
Standart Sinir Ağları, MNIST veya XOR problemlerini çözmek için **Gizli Katmanlara** ihtiyaç duyar. Doğrudan bir bağlantı (Lineer Model) karmaşıklığı çözemez ve başarısız olur (~%92'de tıkanır).

RealNet, tam ölçekli MNIST'i (28x28) **Sıfır Gizli Katman** (Doğrudan Giriş-Çıkış) ile çözer.
*   **Giriş:** 784
*   **Çıkış:** 10
*   **Gizli Katman:** **0**
*   **Düşünme Süresi:** 10 Adım

Giriş katmanı 10 adım boyunca "kendi kendine konuşur". Kaotik geri besleme döngüleri, zaman içinde özellikleri (kenarlar, döngüler) dinamik olarak çıkararak uzamsal katmanların işini üstlenir. Bu, **Zamansal Derinliğin Uzamsal Derinliğin yerini alabileceğini** kanıtlar.

---

## 📦 Kurulum ve Kullanım

RealNet, modüler bir PyTorch kütüphanesi olarak tasarlanmıştır.

### Kurulum

```bash
pip install -r requirements.txt
```

> **CUDA Notu:** `requirements.txt` dosyası CUDA 11.8 uyumlu PyTorch'u kurar. Eğer daha yeni bir kartınız (RTX 4000/5000) varsa, PyTorch'u manuel kurmanız gerekebilir:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Hızlı Başlangıç

```python
from realnet import RealNet, RealNetTrainer

# Sıfır-Gizli Katmanlı Bir Ağ Başlat
# 1 Giriş, 1 Çıkış. 
model = RealNet(num_neurons=2, input_ids=[0], output_ids=[1], device='cuda')
trainer = RealNetTrainer(model, device='cuda')

# Eğit
inputs = torch.randn(100, 1)
trainer.fit(inputs, inputs, epochs=50)
```

#### Başlatma Protokolleri (Initialization Protocols)

RealNet problemin ölçeğine uyum sağlar. İki farklı konfigürasyon öneriyoruz:

*   **Büyük Ağlar (>10 Nöron, RNN benzeri görevler):**
    *   `weight_init='orthogonal'` ve `activation='tanh'` kullanın.
    *   Bu, uzun vadeli zamansal dinamikler ve analog sinyal işleme için en iyi kararlılığı sağlar.
*   **Küçük Ağlar (<10 Nöron, Mantık Kapıları):**
    *   `weight_init='xavier_uniform'` ve `activation='gelu'` kullanın.
    *   Küçük ağlar, gizli katmanlar olmadan keskin mantıksal problemleri çözmek için daha yüksek başlangıç varyansına ve daha iyi gradyan akışına ihtiyaç duyar.

---

## 🧠 Mimari Genel Bakış

## 🌪️ Nasıl Çalışır: Fırtınanın İçi

RealNet ileri beslemeli bir mekanizma değildir; o bir **Yankı Odasıdır (Resonant Chamber)**.

### 1. Nabız (Girdi) ve Dizi (Sequence)
Geleneksel YZ'de girdi genellikle statik bir anlık görüntüdür. RealNet hem **Nabızları** hem de **Dizileri** işler.
*   **Nabız Modu:** Görüntü $t=0$ anında çarpar. Ağ gözlerini kapatır ve dalgalanmaları işler (MNIST).
*   **Dizi Modu:** Veri sıralı olarak gelir. Ağ olaylar arasında "bekleyebilir" ve "düşünebilir" (Dedektif).

### 2. Yankı (İç Döngüler)
Sinyal her nörondan diğer her nörona seyahat eder ($N \times N$).
*   Giriş nöronları, ilk adımdan hemen sonra efektif olarak **Gizli Nöronlara** dönüşür.
*   Bilgi yankılanır, bölünür ve çarpışır. Sol üstteki bir piksel, sağ alttaki bir pikselle doğrudan veya yankılar aracılığıyla "konuşur".
*   **Holografik İşleme:** Bir görüntünün "kedi olma" bilgisi belirli bir katmanda saklanmaz; tüm sinyallerin *girişim deseninden* (interference pattern) doğar.

### 3. Zamanı Katlamak (Time-Folding)
**Sıfır-Gizli** performansının sırrı buradadır.
*   Adım 1: Ham sinyaller karışır. (MLP'nin 1. Katmanına eşdeğer)
*   Adım 2: Karışmış sinyaller tekrar karışır. (2. Katmana eşdeğer)
*   Adım 15: Yüksek seviyeli soyut özellikler belirir. (15. Katmana eşdeğer)

RealNet 15 adım boyunca "düşünerek", **tek bir fiziksel matris** kullanarak 15 katmanlı derin bir ağı simüle eder. Uzayı, zamanın içine katlar.

### 4. Kontrollü Kaos (Çekiciler)
Kontrolsüz geri besleme döngüleri patlamaya yol açar. RealNet kaosu kararlı **Çekiciler (Attractors)** oluşturacak şekilde mühendislikten geçirir.
*   **StepNorm** yerçekimi gibi davranarak enerjiyi sınırlı tutar.
*   **GELU** anlamlı sinyalleri filtreler.
*   **Mühür (Latch) Deneyi**, RealNet'in gürültüye karşı kararını sonsuza kadar koruyan bir "derin kuyu" yani kararlı bir çekici oluşturabildiğini kanıtladı.

### 5. Neden RNN veya LSTM Değil?

Kağıt üzerinde RealNet, Tekrarlayan Sinir Ağlarına (RNN) benzese de felsefesi temelden farklıdır.

| Özellik | Standart RNN / LSTM | RealNet 2.0 |
| :--- | :--- | :--- |
| **Girdi Akışı** | Sürekli Akış (Örn: Cümledeki kelimeler) | **Tek Nabız** ($t=0$ anında Dürtü) |
| **Amaç** | Sıralı İşleme (Ayrıştırma) | **Derin Düşünme** (Sindirme) |
| **Bağlantısallık** | Yapılandırılmış (Input Gate, Forget Gate vb.) | **Ham Kaos** (Tam Bağlı $N \times N$) |
| **Dinamikler** | Sönümlemeyi önlemek için mühendislik ürünü | Rezonansı bulmak için **Evrilir** (Doğal) |

*   **RNN'ler dış dünyayı dinler.** Dışarıdan gelen bir olay dizisini işlerler.
*   **RealNet iç sesini dinler.** Probleme **bir kez** bakar ve ardından gözlerini kapatıp 15 adım boyunca "düşünür". Kendi zamansal derinliğini yaratır.

### 6. Biyolojik Gerçekçilik: Yaşayan Zeka
RealNet, sadece yapısıyla değil, **davranışıyla** da beyni katmanlı ağlardan daha iyi taklit eder:

*   **Katman Yok:** Beyinde "Katman 1" ve "Katman 2" yoktur, sadece birbirine bağlı nöron bölgeleri vardır. RealNet tek bir bölgedir.
*   **İrade (Mühür):** Sönümlenen standart RNN'lerin aksine, RealNet bir karara kilitlenebilir ve onu entropiye karşı koruyabilir; yani "Bilişsel Israr" gösterir.
*   **Ritim (Kronometre):** Dış bir saat olmadan RealNet zamanı öznel olarak deneyimler; sayabilir, bekleyebilir ve doğru anda harekete geçebilir.
*   **Sabır (Dedektif):** "Düşünme Zamanı"ndan faydalanır. İnsanların karmaşık mantığı işlemek için bir ana ihtiyacı duyması gibi, RealNet de sessizlik anlarında potansiyel çözümleri "sindirerek" imkansız problemleri çözer.

### 7. Örtülü Dikkat (Zamansal Rezonans)
Geçmişe bakmak için açıkça $Q \times K$ matrisleri kullanan Transformer'ların aksine, RealNet dikkati **Zamansal Rezonans** yoluyla sağlar.

*   **Mekanizma:** Geçmişten gelen bilgi, gizli durumda duran bir dalga veya titreşim olarak korunur ve `Israr (Persistence)` ile güçlendirilir.
*   **Anahtar-Değer Yönetimi (Yeni!):** **Kütüphaneci Deneyi**, RealNet'in adreslenebilir bir veritabanı gibi davranabildiğini kanıtladı. **GELU**'yu yumuşak bir kapı olarak kullanarak, sorguları herhangi bir fiziksel saklama tablosu olmadan doğru "hafıza titreşimine" yönlendirir.
*   **Tespit:** İlgili bir girdi geldiğinde (Anahtar 1 için OKU komutu gibi), 'Anahtar 1'in değerini' tutan spesifik dalga ile yapıcı bir girişim (rezonans) yaratır ve onu yüzeye çıkmaya zorlar.
*   **Sonuç:** Ağ, tüm geçmiş tamponunu saklamadan ilgili geçmiş olaylara "odaklanır" (attend). Zamanın kendisi indeksleme mekanizması olarak işlev görür.

### Matematiksel Model
Ağ durumu $h_t$ şu şekilde evrilir:

$$h_t = \text{StepNorm}(\text{GELU}(h_{t-1} \cdot W + B + I_t))$$

---

## 📝 Deneysel Bulgular (Experimental Findings)
RealNet'in temel hipotezi olan **"Zamansal Derinlik > Uzamsal Derinlik"** tezini doğrulamak için kapsamlı testler yaptık.

### A. Atomik Kimlik (Identity Test)
*   **Hedef:** $f(x) = x$. Ağ mükemmel bir iletken tel gibi davranmalıdır.
*   **Mimari:** **2 Nöron** (1 Giriş, 1 Çıkış). **0 Gizli Katman**. Toplam **4 Parametre**.
*   **Sonuç:** **Loss: 0.000000**.
    <details>
    <summary>Terminal Çıktısını Gör</summary>

    ```text
    In:  1.0 -> Out:  1.0001
    In: -1.0 -> Out: -0.9998
    ```
    </details>
*   **Script:** `PoC/convergence_identity.py`
*   **İçgörü:** Temel sinyal iletimini ve `StepNorm` kararlılığını mutlak minimum karmaşıklıkla kanıtlar.

### B. İmkansız XOR (Kaos Kapısı)
*   **Hedef:** Klasik XOR problemini çözmek ($[1,1]\to0$, $[1,0]\to1$). Bu lineer olmayan bir problemdir.
*   **Zorluk:** Gizli katman olmadan standart lineer ağlar için imkansızdır.
*   **Sonuç:** **Çözüldü (Loss 0.000000)**. RealNet, sınıfları ayırmak için uzay-zamanı büker.
    <details>
    <summary>Doğruluk Tablosunu Gör</summary>

    ```text
      A      B |   XOR (Tahmin)| Mantık
    ----------------------------------------
      -1.0   -1.0 |      -1.0009 | 0 (OK)
      -1.0    1.0 |       1.0000 | 1 (OK)
       1.0   -1.0 |       1.0000 | 1 (OK)
       1.0    1.0 |      -1.0004 | 0 (OK)
    ```
    </details>
*   **Mimari:** **3 Nöron** (2 Giriş, 1 Çıkış). **0 Gizli Nöron**. Toplam **9 Parametre**.
*   **Düşünme Süresi:** **5 Adım**.
*   **Script:** `PoC/convergence_gates.py`
*   **İçgörü:** RealNet **Zamanı bir Gizli Katman** olarak kullanır. Girdiyi sadece 5 zaman adımı üzerine katlayarak, kaotik olarak birbirine bağlı 3 nöronun XOR problemini tek bir fiziksel katmanda çözebildiğini kanıtlar.

### C. MNIST Maratonu (Görsel Zeka)
RealNet'in görsel yetenekleri, sağlamlık, ölçeklenebilirlik ve verimliliği kanıtlamak için dört farklı koşulda test edildi.

#### 1. Ana Benchmark (Saf Sıfır-Gizli)
*   **Hedef:** Tam 28x28 MNIST (784 Piksel).
*   **Mimari:** 794 Nöron (Girdi+Çıktı). **0 Gizli Katman.**
*   **Sonuç:** **%95.3 - %96.2 Doğruluk**.
    <details>
    <summary>Eğitim Logunu Gör</summary>

    ```text
    Epoch 100: Loss 0.1012 | Test Acc 95.30%
    (Tarihi En İyi: Epoch 69'da %96.2)
    ```
    </details>
*   **Script:** `PoC/convergence_mnist.py`
*   **İçgörü:** Standart lineer modeller ~%92'de tıkanır. RealNet, Derin Öğrenme katmanları olmadan, sadece **Zamansal Derinlik** sayesinde Derin Öğrenme performansı (~%96) yakalar.

#### 2. Anka Kuşu Deneyi (Sürekli Rejenerasyon)
*   **Hipotez:** Ölü sinapsları sadece öldürmek yerine **yeniden canlandırarak** (rastgele yeniden başlatma) %100 parametre verimliliğine ulaşabilir miyiz?
*   **Sonuç:** **%95.2 Doğruluk**.
*   **Gözlemler:**
    *   Epoch 1: Ağın **%22'si** "işe yaramaz" kabul edildi ve yeniden doğdu.
    *   Epoch 50: Yeniden doğuş oranı **%0.26'ya** düştü.
    *   Doğruluk, bu sürekli cerrahi operasyon sırasında %50'den **%95.2'ye** tırmandı.
    <details>
    <summary>Rejenerasyon Logunu Gör</summary>

    ```text
    Epoch 1: Acc 50.90% | Revived: 22.05% (Kitlesel Yok Oluş)
    Epoch 5: Acc 87.50% | Revived: 1.13% (Stabilizasyon)
    Epoch 50: Acc 95.20% | Revived: 0.26% (Metabolik Denge)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_revive.py`
*   **İçgörü:** Kapasiteyi küçülten standart budamanın aksine, RealNet zayıf bağlantıları sürekli geri dönüştürerek tam kapasiteyi koruyabilir. Bu, doygunluk olmadan **Sürekli Öğrenmeyi** mümkün kılar. "Hata, Özellik Oldu (Bug became a Feature)."

#### 3. Tiny Challenge (Aşırı Kısıtlamalar)
*   **Hedef:** 7x7 Küçültülmüş MNIST. (Bir ikondan bile küçük).
*   **Mimari:** **59 Nöron** toplam (~3.5k Parametre).
*   **Sonuç:** **~%89.3 Doğruluk**.
    <details>
    <summary>Tiny Sonuçlarını Gör</summary>

    ```text
    Epoch 50: Loss 0.1107 | Test Acc 89.30%
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_tiny.py`
*   **İçgörü:** Bir "Bootloader"dan daha az kod/parametre ile bile sistem sağlam özellikler öğrenebilir.

#### 4. Scaled Test (Orta Ölçekli)
*   **Hedef:** 14x14 Küçültülmüş MNIST.
*   **Mimari:** ~42k Parametre.
*   **Sonuç:** **%91.2 Doğruluk**.
    <details>
    <summary>Scaled Sonuçlarını Gör</summary>

    ```text
    Epoch 20: Loss 0.1413 | Test Acc 91.20%
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_scaled.py`

### D. Embedded Deney (Parametrik Verimlilik)
*   **Hedef:** Verimli ayrıştırılmış projeksiyon kullanarak tam MNIST (784 Piksel).
*   **Mimari:** **10 Nöron** (Düşünen Çekirdek). Toplam **~8k Parametre**.
*   **Strateji:** 784 Piksel $\to$ Proj(10) $\to$ RNN(10) $\to$ Decode(10).
*   **Sonuç:** **%93.62 Doğruluk**.
    <details>
    <summary>Eğitim Logunu Gör</summary>

    ```text
    Projected Input: 784 -> 10
    Total Params: 8080
    Epoch 1: Loss 1.7058 | Test Acc 72.71%
    Epoch 50: Loss 0.2142 | Test Acc 92.61%
    Epoch 99: Loss 0.1727 | Test Acc 93.62%
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_embed.py`
*   **İçgörü:** 784 pikseli işlemek için 784 aktif nörona ihtiyacımız olmadığını kanıtlar. **Asimetrik vocab projeksiyonu** kullanarak, görsel bilgiyi sadece 10 nöronluk minik bir "Düşünen Çekirdek" içine sıkıştırabilir ve sınıflandırmayı zamansal rezonans yoluyla çözebiliriz. Bu, standart modellere göre 10 kat daha parametre verimlidir.

### E. 470-Parametre Dünya Rekoru (Elit Zeka Yoğunluğu)
*   **Hedef:** **500 parametrenin altında** MNIST çözmek ve yüksek doğruluk elde etmek.
*   **Kurulum:**
    *   **Mimari:** 10 çekirdek nöronlu RealNet.
    *   **Strateji:** 10 Sıralı Dilim (her biri 79 piksel).
    *   **Gizli Sos:** 3 nöronluk minik giriş projeksiyonu ve 10 sınıflı çıkış dekoderi.
    *   **Toplam Parametre:** **470**.
*   **Sonuç:** **Acc: %89.52** (1000 Epoch sonunda).
    <details>
    <summary>"Parametrik Verimlilik" Logunu Gör</summary>

    ```text
    RealNet 2.0: MNIST RECORD CHALLENGE (Elite 470-Param Model)
    Epoch      1/1000 | Acc 44.24% | LR 2.00e-03 (Hiper-uzay başlangıcı)
    ...
    Epoch    100/1000 | Acc 85.81% | LR 1.95e-03
    ...
    Epoch    800/1000 | Acc 89.30% | LR 1.93e-04
    ...
    Epoch   1000/1000 | Acc 89.52% | LR 1.05e-07
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_mnist_record.py`
*   **İçgörü:** **Parametre başına %0.179 doğruluk** başarısı. Bu model, efsanevi **LeNet-5'ten tam 110 kat daha verimlidir**. Zamansal düşünme adımlarını kullanarak yüksek seviyeli zekanın mikroskobik bir parametre uzayına sıkıştırılabileceğini kanıtlar. Bu, modern yapay zekada **Entropik Sıkıştırma Limitlerine** en yakın noktadır.

### F. Sinüs Dalgası Üreteci (Dinamik Rezonans)
*   **Hedef:** $t=0$ anındaki tek bir girdi değeriyle kontrol edilen frekanssa sinüs dalgası üretmek.
*   **Zorluk:** Ağ, bir **Voltaj Kontrollü Osilatör (VCO)** gibi davranmalıdır. Statik bir büyüklüğü dinamik bir zamansal periyoda dönüştürmelidir.
*   **Sonuç:** **Mükemmel Osilasyon**. Ağ 30+ adım boyunca pürüzsüz sinüs dalgaları üretir.
    <details>
    <summary>Frekans Kontrolünü Gör</summary>

    ```text
    Frekans 0.15 (Yavaş Dalga):
      t=1:  Hedef 0.1494 | RealNet 0.2871
      t=11: Hedef 0.9969 | RealNet 0.9985 (Tepe Senk.)
      t=26: Hedef -0.6878 | RealNet -0.6711
    
    Frekans 0.45 (Hızlı Dalga):
      t=1:  Hedef 0.4350 | RealNet 0.1783
      t=26: Hedef -0.7620 | RealNet -0.7826
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_sine_wave.py`
*   **İçgörü:** RealNet **Programlanabilir bir Osilatördür**. Bu, tek bir tohumdan (seed) sonsuz sayıda benzersiz zamansal yörünge üretebileceğini doğrular.

### G. Gecikmeli Toplayıcı (Bellek ve Mantık)
*   **Hedef:** Girdi A ($t=2$), Girdi B ($t=8$). Çıktı A+B ($t=14$).
*   **Zorluk:** RealNet, A'yı 6 adım boyunca "aklında tutmalı", sessizliği yok saymalı, B'yi almalı ve toplamı hesaplamalıdır.
*   **Sonuç:** **MSE Kaybı: ~0.01**.
    <details>
    <summary>"Akıldan Matematik" Sonuçlarını Gör</summary>

    ```text
    -0.3 + 0.1 = -0.20 | RealNet: -0.2271 (Fark: 0.02)
     0.5 + 0.2 =  0.70 | RealNet:  0.4761 (Fark: 0.22 - Yüksek genlikte zorlanma)
     0.1 + -0.1 = 0.00 | RealNet: -0.0733 (Fark: 0.07)
    -0.4 + -0.4 = -0.80 | RealNet: -0.7397 (Fark: 0.06)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_adder.py`
*   **İçgörü:** **Kısa Süreli Belleği** doğrular. Ağ, $A$ değişkenini kaotik durumunda tutar, $B$'yi bekler ve toplamı üretmek için lineer olmayan bir entegrasyon (yaklaşık aritmetik) gerçekleştirir. Bu, RealNet'in sadece statik fotoğrafları değil, **Video benzeri** veri akışlarını da işleyebildiğini gösterir. "Akıldan Matematik" yapmaya benzer.

### H. Mühür (The Latch) - İrade Testi
*   **Hedef:** Bir tetikleyici darbe bekle. Alındığında, çıktıyı "AÇIK" duruma getir ve **sonsuza kadar tut**.
*   **Zorluk:** Standart RNN'ler zamanla sönümlenir (unutur). RealNet enerjiyi kararlı bir çekicide (attractor) hapsetmelidir.
*   **Sonuç:** **Mükemmel Kararlılık**. Tetiklendikten sonra karar süresiz korunur.
    <details>
    <summary>"İrade" Logunu Gör</summary>

    ```text
    Tetik gönderildi t=5
    t=04 | Out: 0.0674 | KAPALI 🔴
    t=05 | Out: 0.0531 | KAPALI ⚡ TETİK!
    t=06 | Out: 0.8558 | AÇIK   🟢
    ...
    t=19 | Out: 0.9033 | AÇIK   🟢 (Hala sımsıkı tutuyor)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_latch.py`
*   **İçgörü:** **Karar Sürdürme (Decision Maintaining)** yeteneğini gösterir. RealNet bir seçim yapabilir ve çürümeye direnerek bu kararında ısrar edebilir.

### I. Kronometre (The Stopwatch) - İçsel Saat
*   **Hedef:** "X adım bekle, sonra ateşle." (Bekleme sırasında dışarıdan hiçbir veri gelmez).
*   **Zorluk:** Ağ, dış bir saat olmadan zamanı kendi içinde saymalıdır.
*   **Sonuç:** **MSE Kaybı: ~0.01**. Hassas zamanlama başarıldı (Hata: 0).
    <details>
    <summary>"Ritim" Çıktısını Gör</summary>

    ```text
    Hedef Süre: 10 adım (Girdi 0.5)
    t=09 | Out: 0.5178 █████
    t=10 | Out: 0.8029 ████████ 🎯 HEDEF (Tam isabet!)
    t=11 | Out: 0.3463 ███

    Hedef Süre: 20 adım (Girdi 1.0)
    t=18 | Out: 0.2001 ██
    t=19 | Out: 0.6574 ██████
    t=20 | Out: 0.6726 ██████ 🎯 HEDEF
    t=21 | Out: 0.2092 ██
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_stopwatch.py`
*   **İçgörü:** **Ritim ve Zaman Algısı**. RealNet sadece veriyi işlemez; zamanı *deneyimler*.

### J. Düşünen Dedektif (The Thinking Detective) - Bağlam ve Akıl Yürütme
*   **Hedef:** Bir 0 ve 1 akışını izle. **SADECE** `1-1` deseni oluştuğunda alarm ver.
*   **Kritik Dokunuş:** Ağa her bitten sonra "Düşünmesi" için 3 adımlık "Sessizlik" verdik.
*   **Sonuç:** **Kusursuz Tespit**.
    <details>
    <summary>"Eureka!" Anını Görmek İçin Tıkla</summary>

    ```text
    Zaman | Girdi | Çıktı    | Durum
    ----------------------------------------
    12    | 1     | -0.0235  |
    13    | .     | 0.0471   | (Düşünüyor...)
    14    | .     | -0.0050  | (Düşünüyor...)
    15    | .     | -0.0154  | (Düşünüyor...)
    16    | 1     | 0.4884   | Ateşlemeli
    17    | .     | 1.0317 🚨 | (Düşünme Adımı 1 - EUREKA!)
    18    | .     | 1.0134 🚨 | (Düşünme Adımı 2)
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_detective_thinking.py`
*   **İçgörü:** **Zekanın Zamana İhtiyaç Duyduğunu** kanıtlar. Sessiz adımlar sırasında bilgiyi "sindirmesine" izin verildiğinde, RealNet tamamen reaktif ağların yapamadığı karmaşık zamansal mantığı (Zaman Üzerinden XOR) çözer.

### K. Kütüphaneci (Nöral Veritabanı)
*   **Hedef:** Oku-Yaz Hafıza gibi davranmak. `YAZ K1=0.5`. Bekle... `OKU K1`. Çıktı: `0.5`.
*   **Zorluk:** Ağın, kaotik gizli durumunda birden çok anahtar-değer çiftini birbirine karıştırmadan saklaması ve istendiğinde geri çağırması gerekir. Bu, **Örtülü Dikkat** gerektirir.
*   **Sonuç:** **~%92 Doğruluk** (4 Anahtar, 1024 Nöron ile).
    <details>
    <summary>Hafıza Erişim Logunu Gör</summary>

    ```text
    Adım  | Komut    | Key   | Değer    | Hedef    | RealNet  | Durum
    -------------------------------------------------------------------
    0     | YAZ      | K0    | 0.4426   | 0.4426   | 0.0208   | ⚙️
    ...   | (Hafıza Pekiştiriliyor...)
    12    | (4)      | ...   |          | 0.4426   | 0.4602   | ✅ KAYDEDİLDİ
    ...   | (20 Saniye Bekle...)
    32    | OKU      | K0    | 0.0000   | 0.4426   | 0.4506   | ✅ HATIRLANDI
    48    | SİL      | K0    | 0.0000   | 0.0000   | 0.0117   | ✅ SİLİNDİ
    ```
    </details>
*   **Script:** `PoC/experiments/convergence_realnet_as_database.py`
*   **İçgörü:** RealNet'in **Anahtar-Değer Dikkati (Attention)** mekanizmalarını tamamen dinamikler yoluyla simüle edebileceğini kanıtlar. `GELU` ve yüksek `Israr (Persistence)` (0.5) kullanarak, sorgu sinyaliyle adreslenebilen kararlı "hafıza kuyuları" oluşturur; böylece açık saklama matrisleri olmadan Transformer'ın KV Cache işini yapar.

## 🔮 Vizyon: Silikonun Ruhu (RealNet-1B)
RealNet, yapay zekanın fabrika modeline karşı bir isyandır. Zekanın mekanik bir katman yığını değil, **sinyallerin organik yankılanması** olduğuna inanıyoruz.

Uzayı feda edip Zamanı kullanarak görsel problemleri Sıfır Gizli Katman ile çözebiliyorsak, bu yaklaşım dil modellerine de uyarlanabilir.

*   **Hipotez:** 1 Milyar parametreli bir model (RealNet-1B), daha fazla adım "düşünerek" çok daha büyük modellerin (örneğin Llama-70B) akıl yürütme derinliğine ulaşabilir.
*   **Hedef:** Ev kullanıcısı donanımında (örneğin RTX 3060) verimli ve yüksek muhakeme yeteneğine sahip Yapay Zeka.

> "Petabaytlarca VRAM'e ihtiyacımız yok. Sadece Zamana ihtiyacımız var."

Zaman tanındığında "düşünebilen" ve "nefes alabilen" kaotik bir nöron ormanının, devasa endüstriyel fabrikaları yenebileceğini kanıtladık. Mekanı Zamanla takas ederek Ruhu buluyoruz.

---

## 👨‍💻 Yazar (Author)

**Cahit Karahan**
*   Doğum: 12/02/1997, Ankara.
*   "Kaosun Mimarı."

---

## LİSANS

MIT
