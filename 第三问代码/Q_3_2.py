import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import time


def calculate_dis(P1,P2):  # 计算P1与P2距离
    p1_arr = np.array(P1)
    p2_arr = np.array(P2)
    result = np.sqrt(np.sum(np.square(p1_arr-p2_arr)))
    return result


def cal_projection(a,b):  # 计算a在b上的投影，输入为a,b向量
    a_arr = np.array(a)
    b_arr = np.array(b)
    result = (np.sum(a_arr * b_arr))/np.linalg.norm(b_arr)
    return result


def data_preprocessing(data1):
    proj_list_for_search = {}
    for j in range(len(data1)):
        proj = cal_projection(data1.iloc[j].values[:3]-data1.iloc[0].values[:3],data1.iloc[-1].values[:3]-data1.iloc[0].values[:3])
        # print(proj)
        proj_list_for_search[j] = proj
    a=[]
    for i in proj_list_for_search.keys():
        a.append(proj_list_for_search[i])
    data1['投影值'] = pd.Series(a)
    proj_list_for_search_sorted = sorted(proj_list_for_search.items(),key = lambda x:x[1],reverse = True)
    reversed_index = []
    for i in range(len(proj_list_for_search_sorted)):
        reversed_index.append(proj_list_for_search_sorted[i][0])
    # 设置成“category”数据类型
    data1['st'] = data1.index
    data1['st'] = data1['st'].astype('category')
    # inplace = True，使 recorder_categories生效
    data1['st'].cat.reorder_categories(reversed_index, inplace=True)
    # inplace = True，使 df生效
    data1.sort_values('st', inplace=True)
    data1.set_index(['st'])
    data1 = data1.drop(['st'], axis = 1)
    # data1 = data1.sort_values(by='投影值',ascending=False)
    return data1


def draw(result_):
    result_.insert(0, [A_x, A_y, A_z])
    result_.insert(100, [B_x, B_y, B_z])
    result_ = list(map(list, zip(*result_)))
    print(result_)
    result_X, result_Y, result_Z = result_[0], result_[1], result_[2]
    X, Y, Z = data_ori.iloc[:, 0], data_ori.iloc[:, 1], data_ori.iloc[:, 2]
    X_ver, Y_ver, Z_ver = X[data_ori["校正点标记"] == 1], Y[data_ori["校正点标记"] == 1], Z[data_ori["校正点标记"] == 1]
    X_hor, Y_hor, Z_hor = X[data_ori['校正点标记'] == 0], Y[data_ori['校正点标记'] == 0], Z[data_ori['校正点标记'] == 0]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_ver, Y_ver, Z_ver, c='b')
    ax.scatter(X_hor, Y_hor, Z_hor, c='y')
    ax1 = fig.gca(projection='3d')
    ax1.plot(result_X, result_Y, result_Z, c='r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig('3_2.png',fig_size = (12,8))
    plt.show()


def greed_solution(data_ori, data, length_ori,b_ax,alpha1, alpha2, beta1, beta2, theta, sigma, K=1):
    data = data.reset_index()
    data['counter'] = range(len(data))
    data_sorted = data.copy()  # 备份排序后的DataFrame
    Ver_pos = data[data["校正点标记"] == 1]  # 获取所有垂直误差校验的DataFrame
    Hor_pos = data[data["校正点标记"] == 0]  # 获取所有水平误差校验的DataFrame
    Ver_pos_ori = Ver_pos.copy()  # 校验点数据的DataFrame，备份一下
    Hor_pos_ori = Hor_pos.copy()
    error_vertical = 0  # 初始化垂直误差为0
    error_horizontal = 0  # 初始化水平误差为0
    tot_length = 0  # 实时保存总长度
    cur = data_sorted.iloc[320]  # 当前在哪个点
    result_list = []  # 保存最终的矫正位置
    global drop_list
    drop_list = []
    count_ver = 0
    count_hor = 0
    data_ = data.loc[(data['Z坐标（单位: m）'] <= 8000) & (data['Z坐标（单位: m）'] >= 3500) & (data['投影值'] > 0)]
    data = data_.copy()
    global success_tag
    success_tag = np.random.choice(a=[0,1],size=1,p=[0.2,0.8])
    for i in range(20):
        # print(data)
        print('data删除后数据大小为：',data.shape)
        # data.to_excel('data1_{0}.xlsx'.format(i))
        Ver_pos_tmp = data[data["校正点标记"] == 1]
        Hor_pos_tmp = data[data["校正点标记"] == 0]
        Ok_Ver_pos_tmp = data[(data["校正点标记"] == 1) & data['第三问点标记']==0]  # 获取所有垂直误差校验的DataFrame
        Prob_Ver_pos_tmp = data[(data["校正点标记"] == 1) & (data['第三问点标记'] == 1)]  # 获取所有垂直误差校验的DataFrame
        Ok_Hor_pos_tmp = data[(data["校正点标记"] == 0) & (data['第三问点标记']==0)]  # 获取所有水平误差校验的DataFrame
        Prob_Hor_pos_tmp = data[(data['校正点标记']==0) & (data['第三问点标记'] == 1)]
        # print(Ok_Hor_pos_tmp.shape,Prob_Hor_pos_tmp.shape)
        # print(Ok_Ver_pos_tmp.shape,Prob_Ver_pos_tmp.shape)
        # print(Ver_pos_tmp.shape,Hor_pos_tmp.shape)
        # if i == 0:
        #     print('Prob ver pos tmp',Prob_Ver_pos_tmp.values)
        Ver_pos_ = Ver_pos_tmp.values[:, :3]  # 获取所有的垂直矫正坐标
        Hor_pos_ = Hor_pos_tmp.values[:, :3]  # 获取所有的水平矫正坐标
        # print('the length of vec {0},hor {1}'.format(len(Ver_pos),len(Hor_pos)))
        # print("tot length {0}".format(tot_length))
        print('当前的节点为：', cur.values)
        print('cur.proj',cur.values[-2])
        if cur.values[-2] >= length_ori:
            print('final proj',cur.values[-2])
            return tot_length, i, result_list
        final_step = calculate_dis(cur.values[1:4],[b_ax.values[0, 0], b_ax.values[0, 1], b_ax.values[0, 2]])
        if final_step <= (theta-error_vertical-1e-4)/0.001 and final_step <= (theta-error_horizontal-1e-4)/0.001:
            print('final step:',final_step)
            print('vec',(theta - error_vertical - 1e-4) / 0.001)
            print('hor', (theta - error_horizontal - 1e-4) / 0.001)
            return tot_length+final_step, i, result_list
        # pos_list1 = []  # 记录满足垂直矫正条件的点
        res_list1 = []  # 记录其对应的距离值
        proj_list1 = Ver_pos_tmp['投影值']  # 保存所有垂直矫正点的投影值
        #         print(proj_list1.values)
        error_vertical_tmp_list = []

        # pos_list2 = []  # 记录满足水平矫正条件的点
        res_list2 = []  # 记录其对应的距离值
        proj_list2 = Hor_pos_tmp['投影值'] # 保存所有水平矫正点的投影值
        # print('proj_list1',proj_list1.to_list())
        error_horizontal_tmp_list = []
        #         print(len(data))
        index_vec_candi = []  # 候选区域内的垂直矫正索引
        index_hor_candi = []  # 候选区域内的水平矫正索引
        print('删掉的长度',len(drop_list))
        cur_list1 = proj_list1
        cur_list2 = proj_list2
        # print('搜索空间的垂直点数 {0} 搜索空间的水平点数 {1}'.format(len(cur_list1),len(cur_list2)))
        index_vec, index_hor = cur_list1.keys().to_list(), cur_list2.keys().to_list()
        index_vec_inter = list(set(index_vec).intersection(set(drop_list)))
        # print('index_vec1',len(index_vec))
        index_vec = [item for item in index_vec if item not in index_vec_inter]
        # print('index_vec2',len(index_vec))
        index_hor_inter = list(set(index_hor).intersection(set(drop_list)))
        # print('index_hor1',len(index_hor))
        index_hor = [item for item in index_hor if item not in index_hor_inter]
        # print('index_hor2',len(index_hor))
        #         print(cur_list1.keys().to_list())
        # print('垂直位置索引',index_vec)
        # print('episode{0}'.format(i))
        # print('第0个数', data_sorted.iloc[0])
        # true_index_vec = []
        for j in range(len(index_vec)):
            # true_index_vec = data[(data['编号'] == index_vec[j])].index
            true_index_vec = index_vec[j]
            # print(true_index_vec)
            # print('1候选点', data_sorted.iloc[true_index_vec].values[:5])
            res1 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_vec].values[1:4])
            error_horizontal_tmp1 = sigma * res1
            error_vertical_tmp1 = sigma * res1
            if (error_horizontal + error_horizontal_tmp1) <= alpha2 and (
                    error_vertical + error_vertical_tmp1) <= alpha1:  # 垂直矫正位置
                if res1 != 0:
                    res_list1.append(res1)
                    error_horizontal_tmp_list.append(error_horizontal_tmp1)
                    index_vec_candi.append(true_index_vec)
        for j in range(len(index_hor)):
            # true_index_hor = data[(data['编号'] == index_hor[j])].index
            true_index_hor = index_hor[j]
            # print(true_index_hor)
            # print('2候选点', data_sorted.iloc[true_index_hor].values[:5])
            res2 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_hor].values[1:4])
            error_horizontal_tmp2 = sigma * res2
            error_vertical_tmp2 = sigma * res2
            if (error_horizontal + error_horizontal_tmp2) <= beta2 and (
                    error_vertical + error_vertical_tmp2) <= beta1:  # 水平矫正位置
                if res2 != 0:
                    res_list2.append(res2)
                    error_vertical_tmp_list.append(error_vertical_tmp2)
                    index_hor_candi.append(true_index_hor)
        # print('index vec candi',index_vec_candi)
        print('满足条件的垂直点数 {0} 满足条件的水平点数 {1}'.format(len(index_vec_candi), len(index_hor_candi)))
        index_vec_candi_K, index_hor_candi_K = index_vec_candi[:K], index_hor_candi[:K]
        res_list1_candi_K, res_list2_candi_K = res_list1[:K], res_list2[:K]
        error_vertical_tmp_list_K,error_horizontal_tmp_list_K = error_vertical_tmp_list[:K],error_horizontal_tmp_list[:K]
        if index_vec_candi_K:  # 如果存在垂直候选位置
            min_res1 = min(res_list1_candi_K)
            # min_index1 = res_list1_candi_K.index(min_res1)
            min_val1 = min_res1
            # print('min_value1', min_val1)
        else:
            min_val1 = 0
            # print('vec point is null')
        if index_hor_candi_K: # 如果存在水平候选位置
            min_res2 = min(res_list2_candi_K)
            # min_index2 = res_list2_candi_K.index(min_res2)
            min_val2 = min_res2
            # print('min_value2', min_val2)
        else:
            min_val2 = 0
        min_val = max(min_val1, min_val2)
        # print('min_value',min_val)
        if min_val == 1e10:
            # print('no points')
            return 0
        if min_val in res_list1_candi_K:  # 垂直校验
            count_ver += 1
            index = res_list1_candi_K.index(min_val)
            index_ori = index_vec_candi[index]
            print('在新的真实的索引位置：',index_ori)
            if index_ori in Prob_Ver_pos_tmp['counter'] and not success_tag:
                print('垂直校验出错了:', Prob_Ver_pos_tmp[Prob_Ver_pos_tmp['counter'] == index_ori])
                print("垂直校验前:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
                error_vertical = min(error_vertical,5)
                error_horizontal += error_horizontal_tmp_list_K[index]
                print("垂直校验后:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            else:
                print('属于正常点')
                print("垂直校验前:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
                error_vertical = 0
                error_horizontal += error_horizontal_tmp_list_K[index]
                print("垂直校验后:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            tot_length += res_list1_candi_K[index]
            result_list.append(data_sorted.iloc[index_ori].values[1:4])  # 将该行保存下来
            cur = data_sorted.iloc[index_ori]  # 将当前业务赋值给该行
            # drop_index = data[data['投影值'] <= cur['投影值']].index
            # drop_index_ = data_sorted.loc[drop_index]['编号']
            # drop_list.extend(drop_index_)  # 当小于等于的加入到删除队列
            # print('drop list',drop_list)
            # print('data:',data)
            for i in range(len(drop_list)):
                data = data[data['编号'] != drop_list[i]]
        elif min_val in res_list2_candi_K:
            count_hor += 1
            index = res_list2_candi_K.index(min_val)  # 水平校验，表示在res_list中第几个位置
            # print("索引为", index)
            index_ori = index_hor_candi_K[index]
            print('真实的新索引为：', index_ori)
            if index_ori in Prob_Hor_pos_tmp['counter'] and not success_tag:
                print('水平校验出错了:', Prob_Hor_pos_tmp[Prob_Hor_pos_tmp['counter'] == index_ori])
                print("水平校验前：垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
                error_horizontal = min(error_horizontal, 5)
                error_vertical += error_vertical_tmp_list_K[index]
                print("水平校验后:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            else:
                print('属于正常点')
                print("水平校验前：垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
                error_horizontal = 0
                error_vertical += error_vertical_tmp_list_K[index]
                print("水平校验后:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            tot_length += res_list2_candi_K[index]
            result_list.append(data_sorted.iloc[index_ori].values[1:4])
            cur = data_sorted.iloc[index_ori]
            # step = min_val
            # print('投影值', cur['投影值'])
            # cur_proj = cur['投影值']

        else:
            print("error!")
            return 0
    return tot_length, i, result_list


if __name__ == '__main__':
    # 说明：E列（校正点类型）：1表示垂直误差校正点，0表示水平误差校正点，
    # F列（第三问标记）：1表示第三问中可能出现问题的点，0表示正常校正点
    data1 = pd.read_excel('data2.xlsx', index_col=0)
    data_ori = data1.copy()
    # 初始化变量值
    alpha1 = 20  # 当垂直误差不大于alpha1,水平误差不大于alpha2时可进行垂直误差校正
    alpha2 = 10
    beta1 = 15  # 当垂直误差不大于beta1,水平误差不大于beta2时可进行水平误差校正
    beta2 = 20
    theta = 20  # 到达终点时，垂直误差和水平误差均应小于theta个单位
    sigma = 0.001  # 飞行器每飞行1m，垂直误差和水平误差各增加sigma个单位
    data1 = data_preprocessing(data1)
    # data1.to_excel('data2_sorted.xlsx')
    A_ax = data1[data1["校正点标记"] == 'A点']
    A_x, A_y, A_z = A_ax.values[0, 0], A_ax.values[0, 1], A_ax.values[0, 2]
    B_ax = data1[data1["校正点标记"] == 'B点']
    B_x, B_y, B_z = B_ax.values[0, 0], B_ax.values[0, 1], B_ax.values[0, 2]
    dis_A_B = calculate_dis([A_x, A_y, A_z], [B_x, B_y, B_z])
    score_list = []
    start = time.time()
    start = time.time()
    length, N, result_list = greed_solution(data_ori,data1,dis_A_B,B_ax,alpha1, alpha2, beta1, beta2, theta, sigma)
    score = 0.5 * 0.0001 * length + 0.5 * N
    print("the time of running time:", time.time() - start)
    print('length',length)
    print('N',N)
    print('result',result_list)
    draw(result_list)
