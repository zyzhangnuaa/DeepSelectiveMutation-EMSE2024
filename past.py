# def minimal_test_suit_op(subject_name, mutated_model_path, predictions_path, kind, num, op):
#     mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
#     all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)
#
#     test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
#     # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
#     test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
#     related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
#     all_mutant_info = {}
#     test_set = set()  # 记录所有杀死的测试输入
#     for mutant in related_class_dict:
#         if mutant.split('_')[1] != op:
#             continue
#         if len(related_class_dict[mutant]) == 0:
#             continue
#         mutant_info = {}
#         kill_class = related_class_dict[mutant]
#         for c in kill_class:
#             # 获取杀死该类别的输入
#             kill_input = set(list(eval(all_mutant_class_input_killing_info[mutant][int(c)])))
#             test_set.update(kill_input)
#             mutant_info[int(c)] = kill_input
#         all_mutant_info[mutant] = mutant_info
#     print(all_mutant_info)
#     print(test_set)
#     print(len(test_set))
#
#     start_time = time.clock()
#     # 构建最小的测试充分集
#     prob = pulp.LpProblem(name='minimal_test_suit', sense=pulp.LpMinimize)
#     x = pulp.LpVariable.dicts('x', list(test_set), lowBound=0, upBound=1, cat=pulp.LpInteger)
#
#     # 目标函数
#     prob += pulp.lpSum([x[i] for i in x])
#
#     # 约束函数
#     for mutant in all_mutant_info.keys():  # {'lenet5_AFRs_0.01_mutated_1': {9: {813, 2129, 2582, 6166, 4761, 9692}}, 'lenet5_AFRs_0.01_mutated_10': {0: {8325}, 3: {2953}}
#         class_input_info = all_mutant_info[mutant]  # {9: {813, 2129, 2582, 6166, 4761, 9692}}
#         kill_class = class_input_info.keys()
#         for c in kill_class:  # 对每个变异体上的每个类都要有一个约束函数， 保证其至少被一个测试输入杀死
#             kill_input = class_input_info[c]  # for 9, kill_input = {813, 2129, 2582, 6166, 4761, 9692}
#             el = {t: 0 for t in test_set}
#             for t in kill_input:
#                 el[t] = 1
#             # 定义约束函数
#             prob += pulp.lpSum([el[i] * x[i] for i in x]) >= 1
#
#     prob.solve()
#
#     elapsed = (time.clock() - start_time)
#     print("running time: ", elapsed)
#
#     select_input = []
#     for v in prob.variables():
#         print(v.name, '=', v.varValue)
#         if v.varValue == 1:
#             select_input.append(int(v.name.split('_')[-1]))
#     print('最小测试充分集的大小为:', pulp.value(prob.objective))
#     print('select_input:', select_input)
#
#     if subject_name == 'mnist' or subject_name == 'lenet5':
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         img_rows, img_cols = 28, 28
#         x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#         x_test = x_test.astype('float32')
#         x_test /= 255
#     else:
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols = 32, 32
#         x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
#         x_test = x_test.astype('float32')
#         x_test /= 255
#     test_x = list()
#     test_y = list()
#     for ind in select_input:
#         test_x.append(x_test[ind])
#         test_y.append(y_test[ind])
#
#     if not os.path.exists(os.path.join('test_suit', kind, subject_name)):
#         try:
#             os.makedirs(os.path.join('test_suit', kind, subject_name))
#         except OSError as e:
#             print('Unable to create folder for analysis results:' + str(e))
#
#     hf = h5py.File(os.path.join('test_suit', kind, subject_name, 'data_' + str(num) + '.h5'), 'w')
#     hf.create_dataset('x_test', data=test_x)
#     hf.create_dataset('y_test', data=test_y)
#     hf.close()

# 为某个变异算子产生的变异体构建测试充分集    不用
# subject_name = 'lenet5'
# mutated_model_path = 'mutated_model_all'
# predictions_path = 'predictions_all'
# kind = 'AFRs_class'
# num = 0
# op = 'AFRs'
# minimal_test_suit_op(subject_name, mutated_model_path, predictions_path, kind, num, op)
