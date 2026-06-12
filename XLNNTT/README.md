\# XLNNTT - Vietnamese Sentiment Analysis Project



\## 1. Clone / Download Project



Đặt project vào thư mục:



```powershell

C:\\Users\\tuanh\\Downloads\\XLNNTT

```



\---



\# 2. Create Virtual Environment



Mở PowerShell tại folder project và chạy:



```powershell

python -m venv .venv

```



\---



\# 3. Activate Virtual Environment



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Nếu activate thành công sẽ thấy:



```powershell

(.venv)

```



\---



\# 4. Install Required Libraries



```powershell

pip install pandas scikit-learn matplotlib openpyxl joblib

```



\---



\# 5. Project Structure



Project structure sau khi setup:



```text

XLNNTT/

│

├── .venv/

│

├── models/

│   ├── logistic\_model.pkl

│   ├── svm\_model.pkl

│   └── tfidf\_vectorizer.pkl

│

├── processed\_data/

│

├── src/

│   ├── train\_baseline.py

│   ├── train\_svm.py

│   ├── compare\_models.py

│   ├── infer.py

│   └── visualize.py

│

├── train\_with\_aug\_raw.csv

│

└── README.md

```



\---



\# 6. Go To Source Folder



```powershell

cd .\\XLNNTT\\src

```



\---



\# 7. Train Logistic Regression Model



```powershell

python train\_baseline.py

```



Expected result:



```text

Accuracy:

0.92...

```



Model sẽ được lưu vào:



```text

../models/logistic\_model.pkl

```



\---



\# 8. Train LinearSVC Model



```powershell

python train\_svm.py

```



Expected result:



```text

Accuracy:

0.94...

```



Model sẽ được lưu vào:



```text

../models/svm\_model.pkl

```



\---



\# 9. Run Real-Time Sentiment Prediction



```powershell

python infer.py

```



Ví dụ:



```text

Nhập review: hàng ổn

Prediction: positive

```



Thoát chương trình:



```text

exit

```



\---



\# 10. Compare Multiple Models



```powershell

python compare\_models.py

```



Ví dụ:



```text

Nhập review: hàng ổn



Logistic Regression:

negative



LinearSVC:

positive

```



\---



\# 11. Visualization



```powershell

python visualize.py

```



Chức năng:



\* Label distribution

\* Confusion matrix

\* Visualization for report/slides



\---



\# 12. Models Used



\## Logistic Regression



\* Classical Machine Learning

\* TF-IDF vectorization

\* Accuracy \~92%



\## LinearSVC



\* Support Vector Machine

\* Better performance on sparse text vectors

\* Accuracy \~94%



\---



\# 13. NLP Pipeline



```text

Raw Review

&#x20;   ↓

Text Cleaning

&#x20;   ↓

TF-IDF Vectorization

&#x20;   ↓

Machine Learning Model

&#x20;   ↓

Sentiment Prediction

```



\---



\# 14. Key Features



\* Vietnamese sentiment analysis

\* TF-IDF vectorization

\* Logistic Regression

\* LinearSVC comparison

\* Real-time inference

\* Error analysis

\* NLP experimentation



\---



\# 15. Common Problems



\## Error: pandas not found



Solution:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\---



\## Error: sklearn not found



Solution:



```powershell

pip install scikit-learn

```



\---



\## Error: file not found



Kiểm tra:



\* đúng folder chưa

\* đúng path chưa

\* file CSV tồn tại chưa



\---



\# 16. Future Improvements



Possible upgrades:



\* PhoBERT

\* Transformer NLP

\* Streamlit UI

\* Aspect-based sentiment analysis



\---



\# 17. Author



Vietnamese NLP Sentiment Analysis Project



