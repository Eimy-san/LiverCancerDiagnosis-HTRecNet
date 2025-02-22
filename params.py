# coding:utf-8
class Params(object):
    lr = 10e-2  # 学习率
    batch_size = 32  # batch大小
    img_size = 224  # 图像尺寸
    use_gpu = True  # 是否使用GPU
    num_workers = 4  # 并行线程数
    num_classes = 2  # 分类类别数
    epoch = 100  # 总训练轮数
    tensorboard_output_dir = "./output/pic_late"  # 日志目录
    weight_output_dir = "./output/weight"   #权重目录
    result_output_dir = "./result"
    print_step = 20  # 输出信息间隔次数
    save_step = 4  # 存储模型间隔轮数
    model_name = 'resnet_cbam_sp'