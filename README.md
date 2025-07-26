# HPCv2 Simulator 🖥️⚡

## 📥 Instalasi

### Prasyarat

- Python 3.8 (BatsimPy tidak kompatibel dengan python versi terbaru)
- Nix (untuk simulasi Batsim di Linux)

### Clone dan Setup Environment

#### LINUX
```bash
git clone https://github.com/RakaSP/HPCv2.git
cd HPCv2
python3.8 -m venv myenv

# aktivasi virtual environment (linux)
source myenv/bin/activate

# install dependensi
pip install batsim-py
pip install --upgrade pandas numpy
pip install evalys

```

#### WINDOWS
```bash
git clone https://github.com/RakaSP/HPCv2.git
cd HPCv2
python3.8 -m venv myenv

# aktivasi virtual environment (windows)
myenv/Scripts/activate

# install dependensi (tidak perlu download module batsim pada windows)
pip install -r requirements_windows.txt
```
## 🐧 Panduan Penggunaan Linux

1. **Simulasi Batsim**

   ```bash
   nix-shell env.nix
   python batsim_simulator_baseline.py
   python batsim_simulator_timeout.py  # dengan timeout policy
   ```

2. **Simulasi CSSP**

   ```bash
   source myenv/bin/activate
   python batsim_hpcv2_baseline.py
   python batsim_hpcv2_timeout.py  # dengan timeout policy
   ```

## 🪟 Panduan Penggunaan Windows

> ⚠️ Hanya simulasi HPCv2 yang bisa dijalankan. Batsim membutuhkan `nix-shell` yang tidak tersedia di Windows.

1. **Aktifkan virtual environment**:

   ```bash
   myenv\Scripts\activate
   ```

2. **Jalankan simulasi HPCv2**:

   ```bash
   python simulator_hpcv2.py
   ```

## 💡 Baseline Policy

Pada simulasi dengan script `simulator_*_baseline.py`, semua node dalam sistem akan selalu aktif (idle/computing).

## ⏲️ Timeout Policy

Script `simulator_*_timeout.py` menggunakan kebijakan timeout, yang secara otomatis mengubah status nodes menjadi switch*off setelah node tersebut dalam status idle selama `X` detik. Nilai `X` adalah batas waktu yang ditentukan dalam timeout policy untuk mencegah node dalam status idle terlalu lama. Nilai `X` atau timeout dapat dimodifikasi pada script `simulator*\*\_timeout.py`

## 💻 Scheduling

Terdapat dua scheduler yang bisa digunakan dalam simulasi:

1. **Easy backfilling Scheduler**
2. **FCFS (First-Come, First-Served) Scheduler**

### Memilih Scheduler pada Batsim-Py

Untuk informasi lebih lanjut tentang bagaimana Batsim bekerja dengan penjadwalan dan cara mengonfigurasi BatsimPy, Anda dapat membaca tutorial lengkapnya di [Dokumentasi Scheduling BatsimPy](https://lccasagrande.github.io/batsim-py/tutorials/scheduling.html)

## 📊 Visualisasi Hasil

Gunakan `visualize_results.ipynb` untuk memvisualisasikan hasil penjadwalan dari file `.csv` yang dihasilkan oleh simulasi. Notebook ini akan menampilkan Gantt chart yang menunjukkan hasil penjadwalan dan alokasi node untuk mengeksekusi seluruh jobs.
