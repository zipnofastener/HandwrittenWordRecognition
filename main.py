import os
import sys
import random
import numpy as np

# データファイルのディレクトリの指定
# 例 os.chdir("C:\\Users\\user\\Desktop\\HandwrittenWordRecognition\\Data")

os.chdir("C:\\Users\\(ユーザ名)\\Desktop\\HandwrittenWordRecognition\\Data")


# 初期化
num = 0
cnt = 0
sig = 0

# ひらがなのラベルをここで決めておく
# ファイル圧縮の際にも用いる
hira = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
hira_info1 = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #「あ」
hira_info2 = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #「い」
hira_info3 = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #「う」
hira_info4 = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #「え」
hira_info5 = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #「お」
hira_info6 = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #「か」
hira_info7 = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0] #「き」
hira_info8 = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0] #「く」
hira_info9 = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0] #「け」
hira_info10 = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0] #「こ」
hira_info11 = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] #「さ」
hira_info12 = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] #「し」
hira_info13 = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] #「す」
hira_info14 = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0] #「せ」
hira_info15 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0] #「そ」
hira_info16 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0] #「た」
hira_info17 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0] #「ち」
hira_info18 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0] #「つ」
hira_info19 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0] #「て」
hira_info20 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] #「と」
hira_info = [hira_info1, hira_info2, hira_info3, hira_info4, hira_info5, hira_info6, hira_info7, hira_info8, hira_info9, hira_info10, hira_info11, hira_info12, hira_info13, hira_info14, hira_info15, hira_info16, hira_info17, hira_info18, hira_info19, hira_info20]

# ロジスティック関数 f(x)=1/(1 + exp(-x)) の実装
# 授業資料のアルゴリズムに準拠しております
def logistic(x, w_kj, w_ji, val, delta_w_kj, delta_w_ji, flag):
    # 中間層の出力と出力層の出力
    yj, yk = [], []

    #初期化
    sig = 0

    # 調整済み，安定化定数(α)と学習定数(η)
    ALPHA = 0.3
    ETA = 0.54

    # yk, yjの計算を実際に行っていく
    # w_kj=中間層の重み，w_ji=出力層の重みとして計算する
    cal1 = np.dot(x, w_kj)
    for i in cal1:
        yj.append(1 / (1 + np.exp(-i)))
    yj = (np.array(yj)).reshape(1, 50)
    w_ji = w_ji.T

    # ここでykの計算
    cal2 = np.dot(yj, w_ji)
    for i in range(20):
        yk.append(1 / (1 + np.exp(-cal2[0][i])))   
    yk = np.array(yk)

    #ykのみの場合にフラグを立てる
    if (flag == 1):
        return yk

    # 出力層の重みの計算
    # reshape(n, m)により，m行n列に整形
    w_ji_out = (((ALPHA*(val - yk) * yk * (1 - yk))).reshape(20, 1))*yj
    delta_w_ji = w_ji_out + ETA * delta_w_ji
    sig = np.zeros([1, 50])
    w_ji = w_ji.T
    for i in range(20):
        sig = sig + (val[i] - yk[i]) * yk[i] * (1 - yk[i]) * w_ji[i]

    # 中間層の重みの計算
    w_ji_med = (ALPHA * yj * (1 - yj)) * sig
    w_ji_med = x.reshape(64, 1) * w_ji_med
    delta_w_kj = w_ji_med + (ETA * delta_w_kj)

    return delta_w_kj, delta_w_ji, yj, yk
    
# ファイルの圧縮
def comp(hira, num, cnt, var):
    # 圧縮したものの格納先
    # blockを8分割して，メッシュ特徴量として格納させる
    block = []
    mesh = []
    #L1 ~ L8のブロックにまず分けていく
    #また，格納先のブロックを用意しておく
    l1,l2,l3,l4,l5,l6,l7,l8 =[],[],[],[],[],[],[],[]

    # ファイルの読み込みを行う
    # 全部で4種類あるので，それぞれ20個のデータ分回せるように配列を工夫する．
    # ここでは，ひらがなファイルが20個あるので0~19で考える -> num < 20:
    while num < 20:
        # 種類別（4つあるのでバージョンを1~4に設定）でファイルの読み込み
        # 最初に定義した00 ~ 19に合わせて，while文で回していく
        if var == 1:
            temp = open('./hira0_'+hira[num]+'L.dat', 'r')
        elif var == 2:
            temp = open('./hira0_'+hira[num]+'T.dat', 'r')
        elif var == 3:
            temp = open('./hira1_'+hira[num]+'L.dat', 'r')
        else:
            temp = open('./hira1_'+hira[num]+'T.dat', 'r')
        for i in temp:
            i = list(i)
            i = [int(n) for n in i if n != '\n']

            # 8bit分を格納する 0~63
            if cnt == 0:
                l1.append(sum(i[0:8]))
                l2.append(sum(i[8:16]))
                l3.append(sum(i[16:24]))
                l4.append(sum(i[24:32]))
                l5.append(sum(i[32:40]))
                l6.append(sum(i[40:48]))
                l7.append(sum(i[48:56]))
                l8.append(sum(i[56:64]))

            # 64bit つまり8回分回す (8x8 -> 64bit)
            else:
                l1[0] = l1[0]+sum(i[0:8])
                l2[0] = l2[0]+sum(i[8:16])
                l3[0] = l3[0]+sum(i[16:24])
                l4[0] = l4[0]+sum(i[24:32])
                l5[0] = l5[0]+sum(i[32:40])
                l6[0] = l6[0]+sum(i[40:48])
                l7[0] = l7[0]+sum(i[48:56])
                l8[0] = l8[0]+sum(i[56:64])
                
            # 64bitまで行ったら，今度は64で割る (memo 黒画素の割合は8/64=0.125となる)
            if cnt+1 == 8:
                block.append(l1[0]/64)
                block.append(l2[0]/64)
                block.append(l3[0]/64)
                block.append(l4[0]/64)
                block.append(l5[0]/64)
                block.append(l6[0]/64)
                block.append(l7[0]/64)
                block.append(l8[0]/64)
                # 最後まで行ったら空配列にして，別バージョンを回すようにする．
                l1,l2,l3,l4,l5,l6,l7,l8 =[],[],[],[],[],[],[],[]
                cnt = 0
            else:
                cnt += 1
            if len(block) == 64:
                mesh.append(block)
                block = []
        num += 1
    return mesh

# 筆記者0 or 1 の学習
def learn_single(mesh, aim, cnt, err_list, aim_val, w_kj, w_ji, delta_w_kj, delta_w_ji):
    while(True):
        for i in mesh:
            aim_val = hira_info[aim]
            i = np.array(i)
            aim_val = np.array(aim_val)
            delta_w_kj, delta_w_ji, yj, yk = logistic(i, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji, flag=0)

            # Δwkj, Δwkiは既に求まっているので
            # 重みを更新する
            w_kj += delta_w_kj
            w_ji += delta_w_ji

            #出力ユニットの平均2乗誤差が指定した値の比較をする
            err_list.append((np.sum((aim_val - yk)**2))/20)
            #100分率
            aim = int(cnt / 100)
            cnt += 1

            #2000個のデータ分行う
            if cnt+1 > 2000:
                aim = 0
                cnt = 0
        err = np.sum(err_list) / len(err_list)
        err_list = []
        print("誤差", err)
        size = 0
        for i in range(2000):
            aim_val = hira_info[int(i/100)]
            x = mesh[i]
            y_k = logistic(x, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji, flag=1)
            #print
            if(np.argmax(y_k)==np.argmax(aim_val)):
                size += 1
        #誤差が，事前に指定した値に達しなければ停止する(memo ある設定値以上である場合は、 処理2.に戻って繰り返す。 設定値より小さい場合は終了。 )
        #err(誤差) < 0.0011~14のとき
        if err < 0.00125:
            return w_kj, w_ji, delta_w_kj, delta_w_ji
        print("正答率", (size/2000)*100, "%")

# 筆記者0 and 1の学習
def learn_multi(mesh, aim, cnt, err_list, aim_val, w_kj, w_ji, delta_w_kj, delta_w_ji):
    while(True):
        for j in mesh:
            for i in j:
                aim_val = hira_info[aim]
                i = np.array(i)
                aim_val = np.array(aim_val)
                delta_w_kj, delta_w_ji, yj, yk = logistic(i, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji, flag=0)
                
                #上記同様，重みの更新　下記も同様．
                w_kj += delta_w_kj
                w_ji += delta_w_ji
                err_list.append((np.sum((aim_val - yk)**2)) / 20)
                aim = int(cnt / 100)
                cnt += 1
                if cnt+1 > 2000:
                    aim = 0
                    cnt = 0
        err = np.sum(err_list) / len(err_list)
        err_list = []
        print("誤差", err)
        size = 0
        for i in range(4000):
            if i < 2000:
                aim_val = hira_info[int(i / 100)]
                x = mesh[0][i]
            else:
                aim_val = hira_info[int((i - 2000) / 100)]
                x = mesh[1][i - 2000]
            y_k = logistic(x, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji, flag=1)
            if(i==1):
                print("任意の「あ」は次の通りです", y_k)
            if(np.argmax(y_k)==np.argmax(aim_val)):
                size += 1
        if err < 0.0012:
            return w_kj, w_ji, delta_w_kj, delta_w_ji
        print("正答率", (size/4000)*100, "%")

# 筆記者0 or 1 の識別
def recog_single(mesh, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji):
    size = 0
    for i in range(2000):
        aim_val = hira_info[int(i/100)]
        x = mesh[i]
        y_k = logistic(x, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji, flag=1)
        if(np.argmax(y_k)==np.argmax(aim_val)):
            size += 1
    return (size/2000) * 100
    
# 筆記者0 and 1の識別 (memo 2000*2 に注意)
def recog_multi(mesh, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji):
    size = 0
    for j in mesh:
        for i in range(2000):
            aim_val = hira_info[int(i/100)]
            x = j[i]
            y_k = logistic(x, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji, flag=1)
            if(np.argmax(y_k)==np.argmax(aim_val)):
                size += 1
    return (size/4000) * 100

#それぞれのファイルの圧縮
mesh1 = comp(hira, num, cnt, var=1)
#print(mesh1)
mesh2 = comp(hira, num, cnt, var=2)
#print(mesh2)
mesh3 = comp(hira, num, cnt, var=3)
#print(mesh3)
mesh4 = comp(hira, num, cnt, var=4)
#print(mesh4)

#初期化
aim = 0
cnt = 0
err_list = []
aim_val = hira_info[aim]

#最初の重みの設定
#np.random.seed(seed=20210802)
w_kj = np.random.randn(64, 50)
w_ji = np.random.randn(20, 50)

#重みを更新
delta_w_kj = 0
delta_w_ji = 0

# T1.筆記者0の学習用データを用いて、ニューラルネットの学習を行なえ。
w_kj, w_ji, delta_w_kj, delta_w_ji = learn_single(mesh1, aim, cnt, err_list, aim_val, w_kj, w_ji, delta_w_kj, delta_w_ji)
T2 = recog_single(mesh1, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)
T3 = recog_single(mesh2, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)
T4 = recog_single(mesh4, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)
# T5.筆記者1の学習用データを用いて、ニューラルネットの学習を行なえ。
w_kj, w_ji, delta_w_kj, delta_w_ji = learn_single(mesh3, aim, cnt, err_list, aim_val, w_kj, w_ji, delta_w_kj, delta_w_ji)
T6 = recog_single(mesh3, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)
T7 = recog_single(mesh2, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)
T8 = recog_single(mesh4, w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)
# T9.筆記者0と筆記者1の学習用データを用いて、ニューラルネットの学習を行なえ。
w_kj, w_ji, delta_w_kj, delta_w_ji = learn_multi((mesh1, mesh3), aim, cnt, err_list, aim_val, w_kj, w_ji, delta_w_kj, delta_w_ji)
T10 = recog_multi((mesh1, mesh3), w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)
T11 = recog_multi((mesh2, mesh4), w_kj, w_ji, aim_val, delta_w_kj, delta_w_ji)

print(T2)
#結果を出力
print("問2:", T2, "問3:", T3, "問4:", T4, "問6:", T6, "問7:", T7, "問8:", T8, "問10:", T10, "問11:", T11)