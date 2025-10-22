import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import gspread
from google.oauth2.service_account import Credentials
import random

print("Analisis Data Universitas Indonesia...")

# Setup Google Sheets conn
def setup_gsheets():
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive',
                 'https://www.googleapis.com/auth/spreadsheets']
        
        creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
        client = gspread.authorize(creds)

        # Coba buka spreadsheet yang sudah ada, atau buat baru
        try:
            sheet = client.open("Data Mining Universitas").sheet1
            print("Berhasil membuka spreadsheet yang sudah ada")
        except gspread.SpreadsheetNotFound:
            # Buat spreadsheet baru jika tidak ada
            sheet = client.create("Data Mining Universitas")
            sheet.share(None, perm_type='anyone', role='writer')  # Optional: buat publik
            sheet = sheet.sheet1
            print("Berhasil membuat spreadsheet baru")
        
        return sheet
    except Exception as e:
        print(f"Error setup Google Sheets: {e}")
        return None

# Fetch data dari API dengan retry mechanism
def get_indonesian_universities():
    print("Mengambil data dari API ...")
    
    # Coba API utama dengan timeout lebih lama
    api_url = "http://universities.hipolabs.com/search?country=Indonesia"
    
    try:
        print(f"Mencoba API: {api_url}")
        response = requests.get(api_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"Berhasil mengambil {len(data)} Universitas dari API")
            
            # Jika dapat data dari API, ambil sample 150 data atau semua jika kurang
            if len(data) >= 150:
                return random.sample(data, 150)
            else:
                print(f"Data dari API hanya {len(data)}, menambah dengan data sample...")
                additional_data = create_additional_universities_sample(150 - len(data))
                return data + additional_data
        else:
            print(f"API response error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"API gagal: {e}")
    
    # Jika API gagal, buat data sample 150 universitas
    print("Menggunakan data sample 150 universitas Indonesia...")
    return create_large_universities_sample()

# Buat data sample tambahan
def create_additional_universities_sample(count):
    base_names = [
        "Universitas", "Institut", "Sekolah Tinggi", "Politeknik", "Akademi",
        "STIE", "STMIK", "STKIP", "STT", "STAI"
    ]
    
    cities = [
        "Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Semarang", "Medan",
        "Makassar", "Malang", "Denpasar", "Palembang", "Padang", "Banjarmasin",
        "Manado", "Samarinda", "Batam", "Bogor", "Surakarta", "Cirebon",
        "Tangerang", "Bekasi", "Depok", "Serang", "Purwokerto", "Magelang"
    ]
    
    specialties = [
        "Teknologi", "Ekonomi", "Ilmu Komputer", "Pertanian", "Kesehatan",
        "Hukum", "Pendidikan", "Manajemen", "Teknik", "Seni", "Sains",
        "Komunikasi", "Psikologi", "Kedokteran", "Farmasi"
    ]
    
    additional_data = []
    
    for i in range(count):
        base_name = random.choice(base_names)
        city = random.choice(cities)
        specialty = random.choice(specialties)
        
        if base_name in ["Universitas", "Institut"]:
            university_name = f"{base_name} {specialty} {city}"
        elif base_name in ["STIE", "STMIK", "STKIP", "STT", "STAI"]:
            university_name = f"{base_name} {city}"
        else:
            university_name = f"{base_name} {city}"
        
        domain_base = "".join([word[0].lower() for word in university_name.split()])[:6]
        if len(domain_base) < 3:
            domain_base = university_name.lower().replace(" ", "")[:6]
        
        if random.random() < 0.8:
            domain_ext = "ac.id"
        else:
            domain_ext = random.choice(["com", "sch.id", "or.id", "web.id"])
        
        domain = f"{domain_base}.{domain_ext}"
        
        has_web = random.random() < 0.9
        web_pages = [f"https://{domain}"] if has_web else []
        
        additional_data.append({
            "name": university_name,
            "country": "Indonesia",
            "domains": [domain],
            "web_pages": web_pages
        })
    
    return additional_data

# Buat data sample besar 150 universitas
def create_large_universities_sample():
    print("Membuat dataset 150 universitas Indonesia...")
    
    base_universities = [
        {"name": "Universitas Indonesia", "country": "Indonesia", "domains": ["ui.ac.id"], "web_pages": ["https://ui.ac.id"]},
        {"name": "Institut Teknologi Bandung", "country": "Indonesia", "domains": ["itb.ac.id"], "web_pages": ["https://itb.ac.id"]},
        {"name": "Universitas Gadjah Mada", "country": "Indonesia", "domains": ["ugm.ac.id"], "web_pages": ["https://ugm.ac.id"]},
        {"name": "Institut Pertanian Bogor", "country": "Indonesia", "domains": ["ipb.ac.id"], "web_pages": ["https://ipb.ac.id"]},
        {"name": "Universitas Airlangga", "country": "Indonesia", "domains": ["unair.ac.id"], "web_pages": ["https://unair.ac.id"]},
        {"name": "Universitas Brawijaya", "country": "Indonesia", "domains": ["ub.ac.id"], "web_pages": ["https://ub.ac.id"]},
        {"name": "Universitas Padjadjaran", "country": "Indonesia", "domains": ["unpad.ac.id"], "web_pages": ["https://unpad.ac.id"]},
        {"name": "Universitas Diponegoro", "country": "Indonesia", "domains": ["undip.ac.id"], "web_pages": ["https://undip.ac.id"]},
        {"name": "Universitas Sebelas Maret", "country": "Indonesia", "domains": ["uns.ac.id"], "web_pages": ["https://uns.ac.id"]},
        {"name": "Universitas Negeri Jakarta", "country": "Indonesia", "domains": ["unj.ac.id"], "web_pages": ["https://unj.ac.id"]},
        {"name": "Universitas Hasanuddin", "country": "Indonesia", "domains": ["unhas.ac.id"], "web_pages": ["https://unhas.ac.id"]},
        {"name": "Universitas Andalas", "country": "Indonesia", "domains": ["unand.ac.id"], "web_pages": ["https://unand.ac.id"]},
        {"name": "Universitas Sriwijaya", "country": "Indonesia", "domains": ["unsri.ac.id"], "web_pages": ["https://unsri.ac.id"]},
        {"name": "Universitas Jember", "country": "Indonesia", "domains": ["unej.ac.id"], "web_pages": ["https://unej.ac.id"]},
        {"name": "Universitas Udayana", "country": "Indonesia", "domains": ["unud.ac.id"], "web_pages": ["https://unud.ac.id"]},
        {"name": "Universitas Bina Nusantara", "country": "Indonesia", "domains": ["binus.ac.id"], "web_pages": ["https://binus.ac.id"]},
        {"name": "Universitas Gunadarma", "country": "Indonesia", "domains": ["gunadarma.ac.id"], "web_pages": ["https://gunadarma.ac.id"]},
        {"name": "Universitas Mercu Buana", "country": "Indonesia", "domains": ["mercubuana.ac.id"], "web_pages": ["https://mercubuana.ac.id"]},
        {"name": "Universitas Telkom", "country": "Indonesia", "domains": ["telkomuniversity.ac.id"], "web_pages": ["https://telkomuniversity.ac.id"]},
        {"name": "Universitas Prasetiya Mulya", "country": "Indonesia", "domains": ["prasetiyamulya.ac.id"], "web_pages": ["https://prasetiyamulya.ac.id"]},
    ]
    
    additional_count = 150 - len(base_universities)
    additional_data = create_additional_universities_sample(additional_count)
    
    all_data = base_universities + additional_data
    
    print(f"Data sample berhasil dibuat: {len(all_data)} universitas")
    print(f"- Base data: {len(base_universities)} universitas terkenal")
    print(f"- Generated data: {len(additional_data)} universitas variasi")
    
    return all_data

# Save data ke google sheets
def save_to_google_sheets(df):
    try:
        sheet = setup_gsheets()
        if sheet is None:
            print("Tidak dapat mengakses Google Sheets")
            return False

        # Clear existing data
        sheet.clear()
        print("Data lama berhasil dihapus")

        # Prepare headers
        headers = ['Nama', 'Negara', 'Domain', 'Domain_Extension', 'Panjang_Nama',
                   'Jumlah_Domain', 'Memiliki_Web', 'Domain_Akademik']
        
        # Insert headers
        sheet.insert_row(headers, 1)
        print("Header berhasil ditambahkan")

        # Prepare data rows
        data_to_insert = []
        for _, row in df.iterrows():
            domain = row['domains'][0] if row['domains'] else ''
            data_to_insert.append([
                str(row['name'])[:100],  # Batasi panjang teks
                str(row['country']),
                str(domain),
                str(row['domain_extension']),
                int(row['name_length']),
                int(row['domain_count']),
                int(row['has_web_page']),
                int(row['is_academic_domain'])
            ])

        # Insert data in batches to avoid timeout
        batch_size = 50
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            sheet.insert_rows(batch, row=2 + i)  # Start from row 2 (after header)
            print(f"Batch {i//batch_size + 1} berhasil disimpan: {len(batch)} records")

        print(f"Total {len(data_to_insert)} records berhasil disimpan ke Google Sheets")
        print(f"URL Spreadsheet: https://docs.google.com/spreadsheets/d/{sheet.spreadsheet.id}")
        return True
        
    except Exception as e:
        print(f"Error menyimpan ke Google Sheets: {e}")
        return False

# Main analyst func
def perform_analysis():
    universities = get_indonesian_universities()

    if len(universities) == 0:
        print("Tidak ada data yang ditemukan. Keluar...")
        return
    
    df = pd.DataFrame(universities)
    print(f"Struktur data: {df.shape[0]} baris x {df.shape[1]} kolom")
    print("Kolom yang tersedia:")
    print(df.columns.tolist())

    # Preprocessing dan Feature Engineering
    print("Membuat fitur untuk analisis...")

    def extract_domain_extension(domain_list):
        if domain_list and len(domain_list) > 0:
            domain = domain_list[0]
            parts = domain.split('.')
            return parts[-1] if len(parts) > 1 else 'unknown'
        return 'unknown'
    
    df['domain_extension'] = df['domains'].apply(extract_domain_extension)
    df['has_web_page'] = df['web_pages'].apply(lambda x: 1 if x and len(x) > 0 else 0)
    df['name_length'] = df['name'].apply(lambda x: len(str(x)))
    df['domain_count'] = df['domains'].apply(lambda x: len(x) if x else 0)

    academic_domains = ['ac.id', 'edu', 'ac']
    df['is_academic_domain'] = df['domain_extension'].isin(academic_domains).astype(int)

    print("Sample data setelah preprocessing:")
    print(df[['name', 'domain_extension', 'name_length', 'is_academic_domain']].head(10))

    print("Menyimpan data ke Google Sheets...")
    save_success = save_to_google_sheets(df)
    if not save_success:
        print("Lanjutkan analisis tanpa Google Sheets...")

    # exploratory data analysis
    print("Analisis Eksplorasi Data...")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    domain_counts = df['domain_extension'].value_counts().head(8)
    domain_counts.plot(kind='bar', color='skyblue')
    plt.title('Domain Extension Terpopuler')
    plt.xticks(rotation=45)
    plt.ylabel('Jumlah Universitas')

    plt.subplot(2, 2, 2)
    plt.hist(df['name_length'], bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.title('Distribusi Panjang Nama Universitas')
    plt.xlabel('Panjang Nama')
    plt.ylabel('Frekuensi')

    plt.subplot(2, 2, 3)
    domain_type_counts = df['is_academic_domain'].value_counts()
    colors = ['lightcoral', 'lightblue']
    labels = ['Non-Akademik', 'Akademik']
    plt.pie(domain_type_counts, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title('Proporsi Domain Akademik')

    plt.subplot(2, 2, 4)
    domain_count_dist = df['domain_count'].value_counts().sort_index()
    domain_count_dist.plot(kind='bar', color='orange')
    plt.title('Jumlah Domain per Universitas')
    plt.xlabel('Jumlah Domain')
    plt.ylabel('Frekuensi')

    plt.tight_layout()
    plt.savefig('university_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # prepare Data untuk Decision Tree
    print("Preparing Data untuk Decision Tree...")

    le = LabelEncoder()
    df['domain_extension_encoded'] = le.fit_transform(df['domain_extension'])

    features = ['name_length', 'domain_count', 'has_web_page', 'domain_extension_encoded']
    target = 'is_academic_domain'
    
    X = df[features]
    y = df[target]

    print(f"Fitur yang digunakan: {features}")
    print(f"Target: {target}")
    print(f"Distribusi kelas: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Data Split (70-30):")
    print(f"Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    dt_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Akurasi Model Decision Tree: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Non-Akademik', 'Akademik']))

    feature_importance = pd.DataFrame({
        'Fitur': features,
        'Importance': dt_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Importance Fitur:")
    print(feature_importance)

    plt.figure(figsize=(20, 12))
    plot_tree(
        dt_model,
        feature_names=features,
        class_names=['Non-Akademik', 'Akademik'],
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True
    )
    plt.title('Decision tree: Prediksi Domain Akademik Universitas Indonesia', fontsize=14)
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Analisis Tambahan - Pola Nama Universitas...")

    keywords = ['universitas', 'institute', 'sekolah', 'politeknik', 'stkip', 'stie', 'stmik']
    keyword_pattern = {}

    for keyword in keywords:
        count = df['name'].str.lower().str.contains(keyword).sum()
        percentage = (count / len(df)) * 100
        keyword_pattern[keyword] = count
        print(f"Pola '{keyword}': {count} universitas ({percentage:.1f}%)")

    try:
        sheet = setup_gsheets()
        if sheet is None:
            print("Tidak dapat mengakses Google Sheets untuk hasil analisis")
            return

        # Cari atau buat worksheet hasil analisis
        try:
            results_sheet = sheet.spreadsheet.worksheet("Hasil_Analisis")
            print("Worksheet Hasil_Analisis ditemukan")
        except gspread.WorksheetNotFound:
            print("Membuat worksheet Hasil_Analisis baru")
            results_sheet = sheet.spreadsheet.add_worksheet(
                title="Hasil_Analisis", 
                rows=100, 
                cols=10
            )

        # Clear dan isi data
        results_sheet.clear()
        results_data = [
            ['METRIK', 'NILAI'],
            ['Total_Universitas', len(df)],
            ['Domain Akademik', f"{df['is_academic_domain'].sum()} ({df['is_academic_domain'].mean()*100:.1f}%)"],
            ['Rata-rata Panjang Nama', f"{df['name_length'].mean():.1f} karakter"],
            ['Universitas dengan Web', f"{df['has_web_page'].sum()} ({df['has_web_page'].mean()*100:.1f}%)"],
            ['Accuracy Model', f"{accuracy*100:.2f}%"],
            [''],
            ['Feature Importance'],
            ['Fitur', 'Importance']
        ]
        
        # Tambahkan feature importance
        for _, row in feature_importance.iterrows():
            results_data.append([row['Fitur'], f"{row['Importance']:.4f}"])
        
        results_data.extend([
            [''],
            ['Pola Nama Universitas'],
            ['Keyword', 'Jumlah']
        ])
        
        # Tambahkan pola nama
        for keyword, count in keyword_pattern.items():
            results_data.append([keyword, count])
        
        # Insert semua data sekaligus
        results_sheet.update('A1', results_data)
        print("Hasil analisis berhasil disimpan ke Google Sheets")
        
    except Exception as e:
        print(f"Error menyimpan hasil analisis: {e}")

    print("\n" + "="*60)
    print("RINGKASAN HASIL ANALISIS DATA MINING")
    print("="*60)
    print(f"Total Dataset: {len(df)} universitas")
    print(f"Domain Akademik: {df['is_academic_domain'].sum()} ({df['is_academic_domain'].mean()*100:.1f}%)")
    print(f"Rata-rata Panjang Nama: {df['name_length'].mean():.1f} karakter")
    print(f"Universitas dengan Web: {df['has_web_page'].sum()} ({df['has_web_page'].mean()*100:.1f}%)")
    print(f"Domain Extension Unik: {df['domain_extension'].nunique()}")
    print(f"Akurasi Model Decision Tree: {accuracy*100:.2f}%")
    print(f"Data Split: 70-30 (Training: {X_train.shape[0]}, Testing: {X_test.shape[0]})")
    print("="*60)

    print("\nContoh Universitas:")
    sample_univ = df[['name', 'domain_extension', 'is_academic_domain']].sample(min(5, len(df)))
    for idx, row in sample_univ.iterrows():
        status = "Akademik" if row['is_academic_domain'] else "Non-Akademik"
        print(f"- {row['name']} ({row['domain_extension']}) - {status}")

    print("\nAnalisis selesai.")

if __name__ == "__main__":
    perform_analysis()