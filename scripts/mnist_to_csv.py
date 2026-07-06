import struct
import csv

def convert_mnist_to_csv(image_filepath, label_filepath, output_filepath, num_samples):
    print(f"'{output_filepath}' oluşturuluyor... ({num_samples} örnek)")
    
    # Dosyaları binary okuma (rb) modunda açıyoruz
    with open(image_filepath, 'rb') as f_img, \
         open(label_filepath, 'rb') as f_lbl, \
         open(output_filepath, 'w', newline='') as f_out:
        
        # Görüntü dosyasının başlığını (Magic Number, Eleman Sayısı, Satır, Sütun) okuyoruz
        # >IIII formatı: Big-Endian formatında 4 adet Integer (4 byte x 4 = 16 byte)
        img_magic, img_num, rows, cols = struct.unpack(">IIII", f_img.read(16))
        
        # Etiket dosyasının başlığını okuyoruz
        lbl_magic, lbl_num = struct.unpack(">II", f_lbl.read(8))
        
        writer = csv.writer(f_out)
        
        # İlk satıra başlıkları (Header) yazıyoruz: label, pixel_0, pixel_1 ... pixel_783
        header = ['label'] + [f'pixel_{i}' for i in range(rows * cols)]
        writer.writerow(header)
        
        # İstediğimiz sayı kadar örneği okuyup CSV'ye yazıyoruz
        for i in range(num_samples):
            # 1 Byte okuyarak etiketi alıyoruz
            label = struct.unpack(">B", f_lbl.read(1))[0]
            
            # 784 Byte okuyarak pikselleri alıyoruz
            pixels = struct.unpack(">" + "B" * (rows * cols), f_img.read(rows * cols))
            
            # Etiketi ve pikselleri tek bir satır olarak yazıyoruz
            writer.writerow([label] + list(pixels))
            
    print(f"Başarılı! Dosya kaydedildi: {output_filepath}\n")

# Eğitim seti için 1000 örnek
convert_mnist_to_csv(
    'data/train-images.idx3-ubyte', 
    'data/train-labels.idx1-ubyte', 
    'data/mnist_train_3000.csv', 
    3000
)

# Test seti için 250 örnek
convert_mnist_to_csv(
    'data/t10k-images.idx3-ubyte', 
    'data/t10k-labels.idx1-ubyte', 
    'data/mnist_test_750.csv', 
    750
)