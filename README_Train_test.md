# EasyOCR

## Train
### 1. Install
Создайте и активируйте виртуальное окружение:
``` bash
python3 -m venv myenv
source myenv/bin/activate
```
Установите зависимости:
``` bash
pip install easyocr
pip install -r requirements.txt
```

### 2. Open folder trainer
- ```all_data``` - папка для хранится данные 
  - ```ru_train_filtered``` - обучаеший данные 
  - ```ru_val``` - тестовые данные
- ```config_files``` - папка для хранение конфигураци
  - ```en_filtered_config.yaml``` - изночальный конфигураци
  - ```custom_data_train.yaml``` - последний используемый конфигураци
- ```saved_models``` - папка для хранение результатов обучение 
  - ```cyrillic_g2.pth``` - изначальная моделька для распознавания ru https://drive.google.com/file/d/1PIywV9_WZqNNfUIk6-bs598fX7OZTcbY/view?usp=sharing
- ```run_train.py``` - кастомный код для обучение 

### 3. Подготовка данных
1. Организация данных
```
├── all_data
│   ├── ru_train_filtered
│   │   └── 10001_1.jpg
│   │         ...
│   │   └── labels.csv
│   └── ru_val
│   │   └── 10002_1.jpg
│   │         ...
│   │   └── labels.csv
```
2. Формат данных

```labels.csv```:
```
filename,words
```
 - filename - имя изображение 
 - words - аннотация 

### 4. Обучение
1. Настройка параметров ```config.yaml```
2. меняем путь к config.yaml в ```run_train.py```
3. Запуск обучения 
``` bash
python run_train.py
```

## Test 
### 1. Open folder EasyOCR
- ```model``` - закидываем обученную модель в эту папку 
- ```user_network``` - перекидываем кофиги обученой модельки
  - ```test_run.py``` - хранится код для тестирование обучение 

Убедитесь что модель(в ```model```) и кофиги(в ```user_network```) одникового имя 

### 2. Меняем пути в коде ```test_run.py```

### 3. Запуск
``` bash
python test_run.py
```

