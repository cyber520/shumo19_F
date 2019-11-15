import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import time
from scipy.interpolate import spline
import turtle

def calculate_dis(P1,P2):  # 计算P1与P2距离
    p1_arr = np.array(P1)
    p2_arr = np.array(P2)
    result = np.sqrt(np.sum(np.square(p1_arr-p2_arr)))
    return result


def cal_projection(a,b):  # 计算a在b上的投影，输入为a,b向量
    a_arr = np.array(a)
    b_arr = np.array(b)
    result = (np.sum(a_arr * b_arr))/np.linalg.norm(b_arr)
    # result = a_arr.dot(b_arr)/np.linalg.norm(b_arr)
    return result


def cal_theta(vec1,vec2):  # 计算vec1与vec2的夹角,输入为向量
    La, Lb = np.sqrt(vec1.dot(vec1)), np.sqrt(vec2.dot(vec2))
    cos_angle = vec1.dot(vec2) / (La * Lb)
    theta_value = np.arccos(cos_angle)
    theta_value = theta_value * 360 / (2 * np.pi)
    return theta_value


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


def graw(result_):
    result_.insert(0, [A_x, A_y, A_z])
    result_.insert(100, [B_x, B_y, B_z])
    result_ = list(map(list, zip(*result_)))
    print(result_)
    result_X, result_Y, result_Z = result_[0], result_[1], result_[2]
    X, Y, Z = data_ori.iloc[:, 0], data_ori.iloc[:, 1], data_ori.iloc[:, 2]
    X_ver, Y_ver, Z_ver = X[data_ori["校正点类型"] == 1], Y[data_ori["校正点类型"] == 1], Z[data_ori["校正点类型"] == 1]
    X_hor, Y_hor, Z_hor = X[data_ori['校正点类型'] == 0], Y[data_ori['校正点类型'] == 0], Z[data_ori['校正点类型'] == 0]
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_ver, Y_ver, Z_ver, c='b')
    ax.scatter(X_hor, Y_hor, Z_hor, c='y')
    ax1 = fig.gca(projection='3d')
    ax1.plot(result_X, result_Y, result_Z, c='r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig('2_1.png',fig_size = (12,8))
    plt.show()


def greed_solution(data_ori, data, length_ori,b_ax,vec_a_b, alpha1, alpha2, beta1, beta2, theta, sigma, K=5):
    data = data.reset_index()
    data['counter'] = range(len(data))
    data_sorted = data.copy()  # 备份排序后的DataFrame
    Ver_pos = data[data["校正点类型"] == 1]  # 获取所有垂直误差校验的DataFrame
    Hor_pos = data[data["校正点类型"] == 0]  # 获取所有水平误差校验的DataFrame
    Ver_pos_ori = Ver_pos.copy()  # 校验点数据的DataFrame，备份一下
    Hor_pos_ori = Hor_pos.copy()
    error_vertical = 0  # 初始化垂直误差为0
    error_horizontal = 0  # 初始化水平误差为0
    tot_length = 0  # 实时保存总长度
    cur = data_sorted.iloc[607]  # 当前在哪个点
    result_list = []  # 保存最终的矫正位置
    # step = 0  # 每一步已确定点的大小res
    serach_space = 1000000
    global drop_list  # 定义全局变量,删除点集合的list
    drop_list = []
    # global cur_proj
    # cur_proj = 0
    count_ver = 0
    count_hor = 0
    global last_cur  # 上一个节点
    last_cur = cur
    global theta_old # 上一个点的theta值
    theta_old = 0
    # 8500,2000更远一点，poj为85329.51605670812
    data_ = data.loc[(data['Z坐标（单位: m）'] <= 8500) & (data['Z坐标（单位: m）'] >= 2000) & (data['投影值'] > 0)]
    data = data_.copy()
    for i in range(13):
        # print(data)
        print('data shape',data.shape)
        # print('data删除后数据大小为：',data.shape)
        print('第{0}次theta old值为:{1}'.format(i,theta_old))
        Ver_pos_tmp = data[data["校正点类型"] == 1]  # 获取所有垂直误差校验的DataFrame
        Hor_pos_tmp = data[data["校正点类型"] == 0]  # 获取所有水平误差校验的DataFrame
        Ver_pos_ = Ver_pos_tmp.values[:, :3]  # 获取所有的垂直矫正坐标
        Hor_pos_ = Hor_pos_tmp.values[:, :3]  # 获取所有的水平矫正坐标
        # print('the length of vec {0},hor {1}'.format(len(Ver_pos),len(Hor_pos)))
        # print("tot length {0}".format(tot_length))
        print('第{0}次的cur:{1}'.format(i, cur.values))
        # print('cur: ',cur)
        if cur.values[-2] >= length_ori:
            print('final proj', cur.values[-2])
            return tot_length, i, result_list
        final_step = calculate_dis(cur.values[1:4],[b_ax.values[0, 0], b_ax.values[0, 1], b_ax.values[0, 2]])
        print('final step:', final_step)
        print('vec', (theta - error_vertical - 1e-4) / 0.001)
        print('hor', (theta - error_horizontal - 1e-4) / 0.001)
        if final_step <= (theta-error_vertical-1e-4)/0.001 and final_step <= (theta-error_horizontal-1e-4)/0.001:
            print('final step:',final_step)
            print('vec',(theta - error_vertical - 1e-4) / 0.001)
            print('hor', (theta - error_horizontal - 1e-4) / 0.001)
            return tot_length, i, result_list
        theta_value1 = []  # 记录满足半径约束的垂直矫正theta值
        res_list1 = []  # 记录其对应的距离值
        proj_list1 = Ver_pos_tmp['投影值']  # 保存所有垂直矫正点的投影值
        # print(proj_list1.values)
        error_vertical_tmp_list1 = []
        error_horizontal_tmp_list1 = []
        theta_value2 = []  # 记录满足半径约束的水平矫正点theta值
        res_list2 = []  # 记录其对应的距离值
        proj_list2 = Hor_pos_tmp['投影值'] # 保存所有水平矫正点的投影值
        # print('proj_list1',proj_list1.to_list())
        error_vertical_tmp_list2 = []
        error_horizontal_tmp_list2 = []
        # print('当前节点为：', cur.values)
        # print('cur type',type(cur))
        #         print(len(data))
        index_vec_candi = []  # 候选区域内的垂直矫正索引
        index_hor_candi = []  # 候选区域内的水平矫正索引
        cur_list1 = proj_list1
        cur_list2 = proj_list2
        # print('搜索空间的垂直点数 {0} 搜索空间的水平点数 {1}'.format(len(cur_list1),len(cur_list2)))
        index_vec, index_hor = cur_list1.keys().to_list(), cur_list2.keys().to_list()
        index_vec_inter = list(set(index_vec).intersection(set(drop_list)))
        # print('index_vec1',len(index_vec))
        index_vec = [item for item in index_vec if item not in index_vec_inter]  # 将属于该类点的drop list删除
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
        print('搜索空间的垂直点数 {0} 搜索空间的水平点数 {1}'.format(len(index_vec), len(index_hor)))
        for j in range(len(index_vec)):  # 对于所有的垂直矫正点
            # print(data.iloc[index_vec[j]].values[:3])
            # print('候选点', data_sorted.iloc[true_index_vec].values[0][1:4])
            if i == 0:  # 如果是第一次飞
                # true_index_vec = data[(data['编号'] == index_vec[j])].index
                true_index_vec = index_vec[j]
                res1 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_vec].values[1:4])
                error_horizontal_tmp1 = sigma * res1
                error_vertical_tmp1 = sigma * res1
                if (error_horizontal + error_horizontal_tmp1) <= alpha2 and (error_vertical + error_vertical_tmp1) <= alpha1:  # 垂直矫正位置
                    if res1 != 0:
                        res_list1.append(res1)
                        error_horizontal_tmp_list1.append(error_horizontal_tmp1)
                        error_vertical_tmp_list1.append(error_vertical_tmp1)
                        index_vec_candi.append(true_index_vec)
            else:
                # true_index_vec = data[(data['编号'] == index_vec[j])].index
                true_index_vec = index_vec[j]
                last_cur_np, cur_np = np.array(last_cur.values[1:4]), np.array(cur.values[1:4])
                vec_temp = cur_np - last_cur_np # 上一步的向量
                x_temp_np = np.array(data_sorted.iloc[true_index_vec].values[1:4])
                vec_x_cur = x_temp_np - cur_np  # 保存当前向量值
                theta_each_step = cal_theta(vec_temp, vec_x_cur)
                # print('第{0}行的theta角为{1}:'.format(true_index_vec, theta_each_step))
                theta_direction = cal_theta(vec_x_cur, vec_a_b)
                # print('第{0}行的垂直点当前向量与ab直线的theta角为{1}:'.format(true_index_vec, theta_direction))
                if i >= 2:
                    if i % 2 == 1:
                        theta_each_step = theta_old + theta_each_step
                    else:
                        theta_each_step = theta_old - theta_each_step
                res1 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_vec].values[1:4])
                Radius_tmp1 = res1 / (2 * np.cos(2 * np.pi * (90 - theta_each_step) / 360))  # s/2cos(theta)
                res1_true = 2 * np.pi * Radius_tmp1 * (2 * theta_each_step / 360)  # 求圆弧的长度
                error_horizontal_tmp1 = sigma * res1_true
                error_vertical_tmp1 = sigma * res1_true
                if (error_horizontal + error_horizontal_tmp1) <= alpha2 and (error_vertical + error_vertical_tmp1) <= alpha1:  # 垂直矫正位置
                    if res1_true != 0:
                        # print('满足先前条件的垂直矫正对应半径为:', Radius_tmp1)
                        # print('第{0}行的垂直点当前向量与ab直线的theta角为{1}:'.format(true_index_vec, theta_direction))
                        if Radius_tmp1 >= 200:  # 半径大于200米
                            # print('满足当前条件的垂直矫正对应半径为:', Radius_tmp1)
                            res_list1.append(res1_true)
                            error_horizontal_tmp_list1.append(error_horizontal_tmp1)
                            error_vertical_tmp_list1.append(error_vertical_tmp1)
                            index_vec_candi.append(true_index_vec)
                            theta_value1.append(theta_each_step)
        for j in range(len(index_hor)): # 对于水平矫正点
            if i == 0:
                # true_index_hor = data[(data['编号'] == index_hor[j])].index
                true_index_hor = index_hor[j]
                # res2 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_hor].values[0][1:4])
                res2 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_hor].values[1:4])
                error_horizontal_tmp2 = sigma * res2
                error_vertical_tmp2 = sigma * res2
                if (error_horizontal + error_horizontal_tmp2) <= beta2 and (error_vertical + error_vertical_tmp2) <= beta1:  # 水平矫正位置
                    if res2 != 0:
                        res_list2.append(res2)
                        error_vertical_tmp_list2.append(error_vertical_tmp2)
                        error_horizontal_tmp_list2.append(error_horizontal_tmp2)
                        index_hor_candi.append(true_index_hor)
            else:
                # true_index_hor = data[(data['编号'] == index_hor[j])].index
                true_index_hor = index_hor[j]
                last_cur_np, cur_np = np.array(last_cur.values[1:4]), np.array(cur.values[1:4])
                hor_temp = cur_np - last_cur_np  # 上一步的向量值
                # x_temp_np = np.array(data_sorted.iloc[true_index_hor].values[0][1:4])
                x_temp_np = np.array(data_sorted.iloc[true_index_hor].values[1:4])
                res2 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_hor].values[1:4])
                hor_x_cur = x_temp_np - cur_np  # 保存当前向量值
                theta_each_step = cal_theta(hor_temp, hor_x_cur)
                theta_direction = cal_theta(hor_x_cur, vec_a_b)
                # print('第{0}行的水平点当前向量与ab直线的theta角为{1}:'.format(true_index_hor, theta_direction))
                if i >= 2:
                    if i % 2 == 1:
                        theta_each_step = theta_old + theta_each_step
                    else:
                        theta_each_step = theta_old - theta_each_step
                res2 = calculate_dis(cur.values[1:4], data_sorted.iloc[true_index_hor].values[1:4])
                Radius_tmp2 = res2 / (2 * np.cos(2 * np.pi * (90 - theta_each_step) / 360))
                res2_true = 2 * np.pi * Radius_tmp2 * (2 * theta_each_step / 360)
                error_horizontal_tmp2 = sigma * res2_true
                error_vertical_tmp2 = sigma * res2_true
                if (error_horizontal + error_horizontal_tmp2) <= beta2 and (error_vertical + error_vertical_tmp2) <= beta1:  # 水平矫正位置
                    if res2_true != 0:
                        # print('满足先前约束的水平矫正对应半径为:', Radius_tmp2)
                        # print('第{0}行的垂直点当前向量与ab直线的theta角为{1}:'.format(true_index_vec, theta_direction))
                        if Radius_tmp2 >= 200:
                            # print('满足当前约束的水平矫正对应半径为:',Radius_tmp2)
                            res_list2.append(res2_true)
                            error_vertical_tmp_list2.append(error_vertical_tmp2)
                            error_horizontal_tmp_list2.append(error_horizontal_tmp2)
                            index_hor_candi.append(true_index_hor)
                            theta_value2.append(theta_each_step)
        # print('index vec candi',index_vec_candi)
        print('满足约束条件的垂直点数 {0} 满足约束条件的水平点数 {1}'.format(len(index_vec_candi), len(index_hor_candi)))
        index_vec_candi_K, index_hor_candi_K = index_vec_candi[:K], index_hor_candi[:K]
        res_list1_candi_K, res_list2_candi_K = res_list1[:K], res_list2[:K]
        theta_value1_candi_K ,theta_value2_candi_K = theta_value1[:K], theta_value2[:K]
        error_vertical_tmp_list1_K, error_horizontal_tmp_list1_K = error_vertical_tmp_list1[
                                                                   :K], error_horizontal_tmp_list1[:K]
        error_vertical_tmp_list2_K, error_horizontal_tmp_list2_K = error_vertical_tmp_list2[
                                                                   :K], error_horizontal_tmp_list2[:K]
        # error_vertical_tmp_list_K,error_horizontal_tmp_list_K = error_vertical_tmp_list[:K],error_horizontal_tmp_list[:K]
        # print('第{0}次前K个投影最大的满足约束的垂直校验点theta值{1}'.format(i,theta_value1_candi_K))
        # print('第{0}次前K个投影最大的满足约束的水平校验点theta值{1}'.format(i,theta_value2_candi_K))
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
            # print('hor point is null')
        min_val = max(min_val1,min_val2)
        if min_val == 1e10:
            print('no points')
            return 0
        if min_val in res_list1_candi_K:  # 垂直校验
            count_ver += 1
            index = res_list1_candi_K.index(min_val)
            error_vertical += error_horizontal_tmp_list1_K[index]
            error_horizontal += error_vertical_tmp_list1_K[index]
            # print("索引为", index)
            print("垂直校验前:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            error_vertical = 0
            print("垂直校验后:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            tot_length += res_list1_candi_K[index]
            index_ori = index_vec_candi[index]
            result_list.append(data_sorted.iloc[index_ori].values[1:4]) # 将该行保存下来
            last_cur = cur # 将该步的点保存下来
            cur = data_sorted.iloc[index_ori]  # 将当前点赋值给该行
            if i >= 1:
                theta_old = theta_value1_candi_K[index]
            drop_index = data[data['投影值']<= cur['投影值']].index
            drop_index_ = data_sorted.loc[drop_index]['编号']
            drop_list.extend(drop_index_)  # 当小于等于的加入到删除队列
            # print('drop list',drop_list)
            # print('data:',data)
            for i in range(len(drop_list)):
                data = data[data['编号'] != drop_list[i]]
        elif min_val in res_list2_candi_K:
            count_hor += 1
            index = res_list2_candi_K.index(min_val)  # 水平校验，表示在res_list中第几个位置
            error_vertical += error_horizontal_tmp_list2_K[index]
            error_horizontal += error_vertical_tmp_list2_K[index]
            print("水平校验前：垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            error_horizontal = 0
            print("水平校验后:垂直误差 {0}, 水平误差 {1}".format(error_vertical, error_horizontal))
            tot_length += res_list2_candi_K[index]
            index_ori = index_hor_candi_K[index]
            # print('索引为：', index_ori)
            result_list.append(data_sorted.iloc[index_ori].values[1:4])
            last_cur = cur  # 将该步的点保存下来
            cur = data_sorted.iloc[index_ori]
            if i >= 1:
                theta_old = theta_value2_candi_K[index]
            drop_index = data[data['投影值']<= cur['投影值']].index
            drop_index_ = data_sorted.loc[drop_index]['编号']
            drop_list.extend(drop_index_)  # 当小于等于的加入到删除队列
            # print('drop list',drop_list)
            # print('data:',data)
            for i in range(len(drop_list)):
                data = data[data['编号'] != drop_list[i]]
        else:
            print("error!")
            return 0
    return tot_length, i, result_list


if __name__ == '__main__':
    # 说明：E列（校正点类型）：1表示垂直误差校正点，0表示水平误差校正点，
    # F列（第三问标记）：1表示第三问中可能出现问题的点，0表示正常校正点
    data1 = pd.read_excel('data1.xlsx', index_col=0)
    data_ori = data1.copy()
    # 初始化变量值
    alpha1 = 25  # 当垂直误差不大于alpha1,水平误差不大于alpha2时可进行垂直误差校正
    alpha2 = 15
    beta1 = 20  # 当垂直误差不大于beta1,水平误差不大于beta2时可进行水平误差校正
    beta2 = 25
    theta = 30  # 到达终点时，垂直误差和水平误差均应小于theta个单位
    sigma = 0.001  # 飞行器每飞行1m，垂直误差和水平误差各增加sigma个单位
    data1 = data_preprocessing(data1)
    # data1.to_excel('data1_sorted.xlsx')
    A_ax = data1[data1["校正点类型"] == 'A 点']
    A_x, A_y, A_z = A_ax.values[0, 0], A_ax.values[0, 1], A_ax.values[0, 2]
    B_ax = data1[data1["校正点类型"] == 'B点']
    B_x, B_y, B_z = B_ax.values[0, 0], B_ax.values[0, 1], B_ax.values[0, 2]
    dis_A_B = calculate_dis([A_x, A_y, A_z], [B_x, B_y, B_z])
    Vec_A_B = np.array([B_x, B_y, B_z]) - np.array([A_x, A_y, A_z])
    print('distance a-b',dis_A_B)
    start = time.time()
    length, N, result_list = greed_solution(data_ori,data1,dis_A_B,B_ax,Vec_A_B,alpha1, alpha2, beta1, beta2, theta, sigma)
    print('N', N)
    print("the time of running time:", time.time() - start)
    print('result_list',result_list)
    print('total length:',length)
    graw(result_list)

