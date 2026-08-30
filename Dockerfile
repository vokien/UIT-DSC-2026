FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt Python và Pip
RUN apt-get update && apt-get install -y python3-pip git wget unzip htop build-essential

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào image
COPY . .

# Chạy lệnh mặc định (có thể bị ghi đè khi BTC chạy script evaluate của họ)
CMD ["python3", "main.py", "--phase", "predict", "--input", "data/public-official.json", "--output", "submission.json"]
