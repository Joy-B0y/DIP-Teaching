import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    warped_image.fill(255)  # 白色背景
    ### FILL: 基于MLS or RBF 实现 image warping
    
    h, w = image.shape[:2]
    src=np.array(source_pts)
    tgt=np.array(target_pts)
    n=src.shape[0]
    A=np.ones((n,n))
    y=np.zeros((n,2))
    for i in range(n):
        A[:,i]=1/(np.sum((tgt-tgt[i])**2,axis=1)+1e3)

    y=src-tgt
    coef=np.linalg.solve(A,y)

    # for i in range(h):
    #     for j in range(w):
    #         x,y=j,i
    #         b=1/(np.sum((tgt-np.array([x,y]))**2,axis=1)+1e3)
    #         newxy=b@coef+np.array([x,y])
    #         newx,newy=newxy[1],newxy[0]
    #         newx=int(np.clip(newx,0,h-1))
    #         newy=int(np.clip(newy,0,w-1))

    #         warped_image[y,x]=image[newx,newy]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 将网格坐标展平
    flat_grid_x = grid_x.flatten()
    flat_grid_y = grid_y.flatten()
    
    # 计算每个像素的新位置
    flat_coords = np.vstack((flat_grid_x, flat_grid_y)).T
    b = 1 / (np.sum((tgt - flat_coords[:, np.newaxis])**2, axis=2) + 1e3)
    new_coords = b @ coef + flat_coords
    
    # 提取新位置的x和y坐标
    new_x = new_coords[:, 1].reshape(h, w)
    new_y = new_coords[:, 0].reshape(h, w)
    
    # 边界处理
    new_x = np.clip(new_x, 0, h - 1).astype(np.int32)
    new_y = np.clip(new_y, 0, w - 1).astype(np.int32)
    
    # 创建变形后的图像
    warped_image[grid_y, grid_x] = image[new_x, new_y]

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
