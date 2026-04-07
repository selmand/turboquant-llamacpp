# TurboQuant

**Aynı GPU'da daha büyük yapay zeka modelleri, daha uzun context — Google'ın TurboQuant KV cache sıkıştırma algoritması ile.**

TurboQuant, LLM çıkarımı sırasındaki en büyük bellek darboğazı olan KV cache'i 16-bit'ten 3-4 bit'e sıkıştırır. Kalite kaybı sıfır. Aynı GPU'da 5 kat daha fazla context. 12GB VRAM'li bir ekran kartı normalde 32K token işlerken, TurboQuant ile 130K+ token işleyebilir.

```
python install.py        # GPU algılar, llama.cpp derler, model indirir
python run.py            # Sunucuyu başlatır
python chat.py           # Terminal sohbet arayüzü
```

## Ne İşe Yarar?

Bir LLM her token ürettiğinde, önceki tüm tokenlar için key/value vektörleri saklar. Bu **KV cache**, context uzadıkça doğrusal büyür ve GPU belleğini hızla doldurur.

TurboQuant bu vektörleri matematiksel olarak optimal bir algoritmaya ile sıkıştırır:

| Durum | KV Cache Boyutu (64K context, 12B model) | 12GB GPU'ya sığar mı? |
|---|---|---|
| FP16 (standart) | ~10.8 GB | ❌ Modele yer kalmaz |
| **TurboQuant 3/4-bit** | **~1.3 GB** | ✅ 4.9 GB bile boş kalır |

**Eğitim gerektirmez. Kalibrasyon gerektirmez. 3.5 bit'te kalite kaybı sıfır.**

## Hızlı Başlangıç

### Gereksinimler
- Python 3.9+
- Git
- İnternet bağlantısı (ilk kurulum için)

CMake, C++ derleyicisi ve CUDA otomatik algılanır, eksikse kurulur.

### Kurulum

```bash
git clone https://github.com/KULLANICI/turboquant.git
cd turboquant
python install.py
```

Kurulum otomatik olarak:
1. Git, CMake, C++ derleyici kontrol eder, eksikse kurar
2. GPU'yu algılar (NVIDIA CUDA / AMD Vulkan / Apple Metal / CPU)
3. llama.cpp TurboQuant fork'unu klonlar ve doğru GPU flag'leri ile derler
4. Hazır katalogdan veya HuggingFace'ten model indirtir
5. Her şeyi `config.json`'a kaydeder

**Windows kısayolu:** `install_and_run.bat` dosyasına çift tıkla.

### Çalıştırma

```bash
python run.py              # Son kullanılan model ile sunucuyu başlat
python run.py --select     # Farklı model seç (HuggingFace arama dahil)
python run.py --ctx 32768  # Context uzunluğunu ayarla
python run.py --port 1234  # Farklı port
python run.py --status     # Mevcut yapılandırmayı göster
```

### Sohbet

```bash
python chat.py             # Sunucu çalışırken terminal sohbeti
```

Sunucu `http://localhost:8080/v1` adresinde OpenAI uyumlu API sunar. Şunlarla da kullanılabilir:
- **curl** / **Python OpenAI SDK** / **Open WebUI** / OpenAI uyumlu herhangi bir client

## Model Seçimi

`python run.py --select` çalıştırınca:

```
  ── Küçük (1-3B) ──
    [ 1]  Llama 3.2 1B Instruct (Q4_K_M)                   0.8G
    [ 2]  Llama 3.2 3B Instruct (Q4_K_M)                   2.0G

  ── Orta (7-9B) ──
    [ 3]  Mistral 7B Instruct v0.3 (Q4_K_M)                4.4G
    [ 4]  Qwen 2.5 7B Instruct (Q4_K_M)                    4.7G
    [ 5]  Llama 3.1 8B Instruct (Q4_K_M)                   4.9G
    [ 6]  Gemma 2 9B Instruct (Q4_K_M)                     5.4G

  ── Büyük (12-24B) — TurboQuant ile 12GB VRAM'e sığar! ──
    [ 7]  Gemma 3 12B Instruct (Q4_K_M)                    8.1G
    [ 8]  Qwen 2.5 14B Instruct (Q4_K_M)                   9.0G
    [ 9]  DeepSeek R1 Distill Qwen 14B (Q4_K_M)            9.0G
    [10]  Mistral Small 3.1 24B Instruct (IQ4_XS)         10.3G

  ── HuggingFace'ten Başka Model İndir ──
    [11]  🔍 HuggingFace'te ara (tarayıcı açılır, model adını yapıştır)
```

11 numarayı seçince tarayıcıda [HuggingFace model arama](https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:1B,max:32B&apps=llama.cpp&sort=downloads) sayfası açılır. Beğendiğin modelin adını kopyala (ör: `bartowski/Qwen2.5-14B-Instruct-GGUF`), terminale yapıştır — TurboQuant en uygun GGUF dosyasını bulup indirir.

**Sadece GGUF formatı desteklenir** (llama.cpp gereksinimi). Uyumsuz formatlar (MXFP4, AWQ, GPTQ, EXL2) otomatik filtrelenir.

## TurboQuant Nasıl Çalışır?

Algoritma (Google Research, ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) iki aşamadan oluşur:

**Aşama 1 — PolarQuant (MSE-optimal sıkıştırma):**
1. Vektörün normunu kaydet
2. Rastgele ortogonal matris ile döndür (Haar dağılımlı, QR ayrıştırma)
3. Döndürme sonrası her koordinat bilinen bir Beta dağılımına uyar
4. Önceden hesaplanmış Lloyd-Max optimal skaler kuantizer ile 3-4 bit'e sıkıştır

**Aşama 2 — QJL (1-bit bias düzeltme):**
1. Rezidüel hesapla: orijinal − yeniden yapılandırılmış
2. Quantized Johnson-Lindenstrauss dönüşümü uygula (1-bit sign)
3. Attention skorunda tarafsız iç çarpım düzeltmesi olarak kullan

**Sonuç:** Koordinat başına 3-4 bit, Shannon bilgi-kuramsal sınırın 2.7 katı içinde, sıfır eğitim/kalibrasyon gereksinimi.

### Bellek Formatı
```
TQ3 (3-bit): 4 byte norm + 48 byte paketlenmiş indeks = 52 byte / 128-boyutlu vektör
TQ4 (4-bit): 4 byte norm + 64 byte paketlenmiş indeks = 68 byte / 128-boyutlu vektör
FP16:        256 byte / 128-boyutlu vektör

Sıkıştırma: TQ3 = 4.9×, TQ4 = 3.8×
```

## GPU Desteği

| GPU | Backend | Otomatik Algılama |
|---|---|---|
| NVIDIA (GTX/RTX/Tesla/A100/H100) | CUDA | ✅ |
| AMD (RX/Radeon) | Vulkan / ROCm | ✅ |
| Intel Arc | Vulkan | ✅ |
| Apple Silicon (M1/M2/M3/M4) | Metal | ✅ |
| GPU yok | CPU | ✅ (yedek) |

## Proje Yapısı

```
turboquant/
├── install.py              # Tek komut kurulum
├── run.py                  # Sunucu başlatıcı + model seçici
├── chat.py                 # Terminal sohbet arayüzü
├── diagnose.py             # Sistem diagnostik aracı
├── test.py                 # Hızlı test
├── config.json             # Otomatik oluşturulan yapılandırma
├── requirements.txt        # Python bağımlılıkları (numpy, scipy)
├── turboquant/             # Algoritma kütüphanesi
│   ├── errors.py           # Hata kodları (TQ-1xx — TQ-8xx) + izin kontrolleri
│   ├── gpu_detect.py       # GPU otomatik algılama (CUDA/Vulkan/Metal/CPU)
│   ├── rotation.py         # Haar ortogonal rotasyon matrisi
│   ├── lloyd_max.py        # Lloyd-Max optimal skaler kuantizer
│   ├── qjl.py              # Quantized Johnson-Lindenstrauss dönüşümü
│   ├── quantizer.py        # TurboQuantMSE + TurboQuantProd algoritmaları
│   ├── mixed_precision.py  # 2.5/3.5-bit karışık hassasiyet modları
│   └── kv_cache.py         # KV cache yöneticisi (residual window'lu)
├── models/                 # İndirilen GGUF modeller
├── install_and_run.bat     # Windows çift tıkla kurulum
├── run.bat / chat.bat      # Windows kısayolları
└── install.sh              # Linux/Mac kurulum
```

## Hata Yönetimi

Her hata bir kod, açıklama ve çözüm önerisi içerir:

```
╔══ HATA TQ-503 ═════════════════════════════════════
║ Port zaten kullanılıyor
║ Port 8080
╠══ ÇÖZÜM ═══════════════════════════════════════════
║ Farklı port dene: python run.py --port 9090
║ Kullanan uygulamayı bul: netstat -ano | findstr :8080
╚════════════════════════════════════════════════════
```

54 hata kodu: gereksinimler (TQ-1xx), GPU (TQ-2xx), derleme (TQ-3xx), model (TQ-4xx), sunucu (TQ-5xx), dosya sistemi/izinler (TQ-6xx), algoritma (TQ-7xx), ağ (TQ-8xx).

Sorun yaşarsan: `python diagnose.py` tam sistem raporu üretir.

## Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `cmake bulunamadı` | Windows: `winget install Kitware.CMake`, sonra terminal'i yeniden aç |
| `GGML_ASSERT hatası` | Model mimarisi desteklenmiyor — farklı model dene |
| Sunucu yavaş / GPU düşük kullanım | Context'i düşür: `python run.py --ctx 16384 --layers 99` |
| Türkçe karakter bozuk | Son sürümdeki `chat.py`'yi kullan (düzeltildi) |
| Port kullanımda | `python run.py --port 9090` |
| Yanlış model açılıyor | `python run.py --select` ile yeni model seç |

## Referanslar

- [TurboQuant Makalesi (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) — Zandieh ve ark., ICLR 2026
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama.cpp TurboQuant Tartışması](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [TheTom/llama-cpp-turboquant Fork](https://github.com/TheTom/llama-cpp-turboquant)

## Lisans

MIT
