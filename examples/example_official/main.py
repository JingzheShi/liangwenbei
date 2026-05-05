import sys
import pandas as pd
import json

# 定义动态导入类函数
def import_class(import_str):
    mod_str,_seq,class_str = import_str.rpartition(".")
    __import__(mod_str)
    return getattr(sys.modules[mod_str],class_str)


if __name__ == "__main__":
    # 测试
    import_str = "mmpc.Predictor.Predictor"
    DynamicClass = import_class(import_str)

    predictor = DynamicClass()
    
    columns={}
    with open("mmpc/config.json", encoding='utf-8') as a:
        columns = json.load(a)
    
    #加载数据
    df = pd.read_parquet('snapshot_sym0_date0_am.parquet')
    syms = df['sym'].unique()
    dates = df['date'].unique()
    predict_result = []
    slide_window = 100
    for sym in syms:
        for date in dates:
            np_data = df[(df['sym'] == sym)&(df['date']==date)]
            np_data = np_data[columns['feature']].copy()
            for index in range(0, len(np_data) - slide_window):
                data = np_data.iloc[index : index + slide_window,:].copy()
                r = predictor.predict([data])
                print(r)  # 使用一个标签时 输出格式为[[1]], 使用3个标签时 输出格式为[[1,2,1]]， 使用5个标签时 输出格式为[[1,2,1,0,1]]
                    

    


