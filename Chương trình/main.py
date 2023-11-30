import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler

#Thu vien cho bo du doan
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Chạy dữ liệu
def load_data(file_path):
    return pd.read_csv(file_path)

# Biểu đồ số lượng bệnh nhân nam và nữ
def sex_count_plot(df):
    sns.countplot(x='sex', hue='sex', data=df, palette="mako_r", legend=False)
    plt.xlabel("Sex (0 = female, 1= male)")
    plt.show()
    
# Tính tỷ lệ bệnh nhân mắc bệnh và không mắc bệnh tim
def percentage_of_heart_disease(df):
    countNoDisease = len(df[df.target == 0])
    countHaveDisease = len(df[df.target == 1])
    print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / len(df.target)) * 100))
    print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / len(df.target)) * 100))

# Tính tỷ lệ bệnh nhân mắc bệnh tim ở nam và nữ
def percentage_of_gender(df):
    countFemale = len(df[df.sex == 0])
    countMale = len(df[df.sex == 1])
    print("Percentage of Female Patients: {:.2f}%".format((countFemale / len(df.sex)) * 100))
    print("Percentage of Male Patients: {:.2f}%".format((countMale / len(df.sex)) * 100))

# Trung bình các đặc điểm ở bệnh nhân mắc và không mắc bệnh tim
def heart_disease_stats(df):
    print(df.groupby('target').mean())

# Tần suất bệnh nhân mắc và không mắc bệnh tim ở từng độ tuổi
def heart_disease_frequency_age(df):
    pd.crosstab(df.age, df.target).plot(kind="bar", figsize=(20, 6))
    plt.title('Heart Disease Frequency for Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('heartDiseaseAndAges.png')
    plt.show()

# Tần suất bệnh nhân mắc và không mắc bệnh tim theo giới tính
def heart_disease_frequency_sex(df):
    pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
    plt.title('Heart Disease Frequency for Sex')
    plt.xlabel('Sex (0 = Female, 1 = Male)')
    plt.xticks(rotation=0)
    plt.legend(["Haven't Disease", "Have Disease"])
    plt.ylabel('Frequency')
    plt.show()
    
# Nhịp tim tối đa của bệnh nhân mắc và không mắc bệnh tim theo độ tuổi
def maximum_heart_rate(df):
    plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
    plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
    plt.legend(["Disease", "Not Disease"])
    plt.xlabel("Age")
    plt.ylabel("Maximum Heart Rate")
    plt.show()
    
# Tần suất người bị và không bị bệnh tim theo slope
def heart_desease_frequency_for_slope(df):
    pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
    plt.title('Heart Disease Frequency for Slope')
    plt.xlabel('The Slope of The Peak Exercise ST Segment ')
    plt.xticks(rotation = 0)
    plt.ylabel('Frequency')
    plt.show()

# Tấn suất người bị và không bị bệnh tim theo lượng đường trong máu
def heart_disease_frequency_FBS(df):
    pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
    plt.title('Heart Disease Frequency According To FBS')
    plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
    plt.xticks(rotation = 0)
    plt.legend(["Haven't Disease", "Have Disease"])
    plt.ylabel('Frequency of Disease or Not')
    plt.show()

# Tần suất người bị và không bị bệnh tim theo loại đau ngực
def heart_disease_frequency_cp(df):
    pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
    plt.title('Heart Disease Frequency According To Chest Pain Type')
    plt.xlabel('Chest Pain Type')
    plt.xticks(rotation = 0)
    plt.ylabel('Frequency of Disease or Not')
    plt.show()
    
def decision_diagram(dtc_model, feature_names):
    plt.figure(figsize=(30, 10))
    plot_tree(dtc_model, feature_names=feature_names, class_names=['0', '1'], filled=True, rounded=True)
    plt.show()
    
#Tạo biến giả, dữ liệu trước khi xử lý
def preprocess_data(df):
    # Mã hóa cột và thêm tiền tố
    a = pd.get_dummies(df['cp'], prefix="cp")
    b = pd.get_dummies(df['thal'], prefix="thal")
    c = pd.get_dummies(df['slope'], prefix="slope")
    
    # Ghép nối dataframe
    frames = [df, a, b, c]
    return pd.concat(frames, axis=1)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
def split_and_scale_data(df):
    # Trích xuất giá trị y từ khung dữ liệu
    y = df.target.values
    
    # Trích xuất giá trị x_data bằng cách bỏ cột "target"
    x_data = df.drop(['target'], axis=1)
    
    # Chia tỷ lệ các giá trị tính năng
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x_data)
    
    # Chia dữ liệu thành tập huấn luyện và tập thử nghiệm
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Chuyển đổi dữ liệu để phù hợp với định dạng đầu vào dự kiến 
    return x_train.T, y_train.T, x_test.T, y_test.T

# Huấn luyện cây dữ liệu
def train_decision_tree(x_train, y_train):
    #Tạo đối tượng với trạng thái chỉ định ngẫu nhiên
    dtc = DecisionTreeClassifier(random_state=42)
    
    #Điều chỉnh trình phân loại, chuyển đổi hàng và cột dữ liệu đầu vào
    dtc.fit(x_train.T, y_train.T)
    return dtc

# Kiểm thử hiệu suất của thuật toán
def test_decision_tree(dtc, x_test, y_test):
    #Tính toán độ chính xác của dự đoán
    acc = dtc.score(x_test.T, y_test.T) * 100
    print("Decision Tree Test Accuracy {:.2f}%".format(acc))
    return acc

if __name__ == "__main__":
    # Load dữ liệu
    file_path = "Heart.csv.csv"
    df = load_data(file_path)

    print()
    print ("""Heart Disease - Classifications (Machine Learning)""")
    print()
    print ("Options: ", "1. Displays hierarchical tables",
           "2. Shows the prediction tree model",
           "3. Make predictions", sep = "\n")
    a = int(input("Input option > "))
    if (a == 1):
        print()
        # Phân tích và hiển thị dữ liệu
        sex_count_plot(df)
        percentage_of_heart_disease(df)
        percentage_of_gender(df)
        heart_disease_stats(df)
        heart_disease_frequency_age(df)
        heart_disease_frequency_sex(df)
        maximum_heart_rate(df)
        heart_desease_frequency_for_slope(df)
        heart_disease_frequency_FBS(df)
        heart_disease_frequency_cp(df)
        # Xử lý và chia dữ liệu 
        df = preprocess_data(df)
        x_train, y_train, x_test, y_test = split_and_scale_data(df)

        # Huấn luyện trình phân loại dữ liệu
        dtc_model = train_decision_tree(x_train, y_train)
        
        print()
        # Kiểm tra trình phân loại và độ chính xác
        accuracy = test_decision_tree(dtc_model, x_test, y_test)
        print()
    elif (a == 2):
        # Xử lý và chia dữ liệu 
        df = preprocess_data(df)
        x_train, y_train, x_test, y_test = split_and_scale_data(df)

        # Huấn luyện trình phân loại dữ liệu
        dtc_model = train_decision_tree(x_train, y_train)
        # Hiển thị mô hình Decision tree
        decision_diagram(dtc_model, df.columns)
        print()
        # Kiểm tra trình phân loại và độ chính xác
        accuracy = test_decision_tree(dtc_model, x_test, y_test)
        print()
    elif (a == 3):
        print("""Please enter the necessary parameters into the file 'heart1.csv' to predict 'target'""")
        print("""Please confirm you have filled in the information with 'YES/Y/yes/y'""")
        b = input("> ")
        lowercase_b = b.lower()
        first_letter = lowercase_b[0]
        if (first_letter == 'y'):
            # Đọc dữ liệu từ file CSV vào DataFrame
            data_train = pd.read_csv("Heart.csv.csv")

            # Tách features và target từ dữ liệu huấn luyện
            X_train = data_train.drop("target", axis=1)
            y_train = data_train["target"]

            # Khởi tạo mô hình Decision Tree và huấn luyện trên dữ liệu huấn luyện
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            # Đọc dữ liệu cần phán đoán
            data_test = pd.read_csv("heart1.csv")

            # Dự đoán kết quả cho dữ liệu mới
            predictions = model.predict(data_test)

            # Hiển thị kết quả dự đoán
            print()
            print("PREDICTED")
            print()
            for i in range(len(data_test)):
                print(f"Patient {i+1}: Target = {predictions[i]}")
            print()