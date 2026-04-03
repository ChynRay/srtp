import pyrealsense2 as rs
import cv2 
import numpy as np
import open3d as o3d

class VisualizerO3D(object): # 可视化类
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd=o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
    
    def render_new_frame(self,verts,texcoords,color,if_first=False,if_downsample=True):
        # 将纹理坐标映射到颜色图像
        # 假设 texcoords 是 [0..1] 范围的坐标，将其映射到颜色图像的尺寸
        texcoords = np.clip(texcoords, 0, 1)  # 确保纹理坐标在合法范围内
        cw, ch = color.shape[1], color.shape[0]  # 获取颜色图像的宽度和高度

        u = (texcoords[:, 0] * (cw - 1)).astype(int)
        v = (texcoords[:, 1] * (ch - 1)).astype(int)

        colors = color[v, u] / 255.0  # 归一化到 [0, 1] 范围
    
        self.pcd.points = o3d.utility.Vector3dVector(verts)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if if_downsample:
            self.downsample()
        if if_first:
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
        else:
            self.vis.reset_view_point(True)
            self.vis.update_geometry(self.pcd)    

        self.vis.poll_events()   
        self.vis.update_renderer()
        print('finish render!')
    
    def downsample(self,uniform_downsample_ratio=5):  # 点云下采样

        self.pcd = self.pcd.uniform_down_sample(every_k_points=uniform_downsample_ratio)
    def shutdown(self):    # 关闭可视化窗口
        self.vis.destroy_window()


class RealSenseCamera:  
    def __init__(self, width=1280, height=720, fps=15):
        """
        Realsense相机控制
        :param width: 相机分辨率宽度
        :param height: 相机分辨率高度
        :param fps: 相机帧率
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.align = None
        self.point_cloud = None
        
    # 开始工作
    def start_work(self):
        # 启动管道
        self.pipeline.start(self.config)
        self.init_threshold_filter()
    def stop_work(self):
        self.pipeline.stop()
        
    def work_flow(self,save_path):
        """
        读取并保存一帧图像
        :param save_path: 保存路径
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        intri=color_frame.profile.as_video_stream_profile().intrinsics
        dist_coeffs=intri.coeffs
        f=intri.fx
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(save_path,color_image)
        return color_image
    def vis_aligned_RGBD(self, depth_image, color_image):
        """
        可视化对齐后的RGBD图像
        :param depth_img: 深度图像
        :param color_img: 彩色图像
        :return: None
        """
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        image_show = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense', image_show)
        cv2.waitKey(1)

    def aligned_RGBD_flow(self):
        """
        获取RGBD图像，每调用一次，返回一组depth_image, color_image

        :return: depth_image, color_image
        """
        # 如果align为空，则调用aligner_enable()函数
        if self.align is None:
            self.aligner_enable()

        # 等待获取帧
        frames = self.pipeline.wait_for_frames()
        # 对齐帧
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        return depth_frame, color_frame

    def frame2numpy(self, frame):
        """
        将帧转换为numpy数组
        :param frame: 帧对象
        :return: numpy数组
        """
        return np.asanyarray(frame.get_data())

    def aligner_enable(self):
        """
        启动RGBD图像对齐器
        """
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # Align the depth frame to color frame
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def point_cloud_enable(self):
        """
        启动点云生成器
        """
        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        # decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
        colorizer = rs.colorizer()

        self.point_cloud = pc
        self.decimate = decimate
        self.colorizer = colorizer

    def init_threshold_filter(self,min_distance=0.2, max_distance=3):
        """
        深度阈值滤波器
        :param depth_frame: 深度帧
        :param min_distance: 最小距离
        :param max_distance: 最大距离
        :return: 阈值滤波后的深度帧
        """
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, min_distance)  # 最小距离（米）
        self.threshold_filter.set_option(rs.option.max_distance, max_distance)
        # return self.threshold_filter.process(depth_frame)

    def undistortion_image(self,color_image):
        """
        图像去畸变
        :param color_image: 原始图像
        :return: 去畸变后的图像
        """
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        fx_color=color_intrinsics.fx
        fy_color=color_intrinsics.fy
        ppx_color=color_intrinsics.ppx
        ppy_color=color_intrinsics.ppy
        distortion=color_intrinsics.coeffs
        cm=np.array([[fx_color,0,ppx_color],[0,fy_color,ppy_color],[0,0,1]])
        undistorted_image = cv2.undistort(color_image, cm, distortion)
        # 可视化去畸变效果
        cv2.imshow('undistorted_image', undistorted_image)
        cv2.waitKey(1)
        return undistorted_image

    def point_cloud_flow(self):
        """
        获取点云，每调用一次，返回一组点云
        """
        if self.point_cloud is None:
            self.point_cloud_enable()
        depth_frame, color_frame = self.aligned_RGBD_flow()

        depth_frame = self.threshold_filter.process(depth_frame)
        color_image = self.frame2numpy(color_frame)
        depth_image= self.frame2numpy(depth_frame)
      
        self.point_cloud.map_to(color_frame)
        points = self.point_cloud.calculate(depth_frame)
          # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        return verts, texcoords, color_image

if __name__ == '__main__':
    import time
    
    camera=RealSenseCamera()
    
    camera.start_work()
    visualizer=VisualizerO3D()
    num=0
    while True:
        if num==0:
            if_first=True
        else:
            if_first=False
        camera.point_cloud_enable()
        verts, texcoords, color_image= camera.point_cloud_flow()
        visualizer.render_new_frame(verts,texcoords,color_image,if_first)
        # time.sleep(2)
        num+=1
    # camera.stop_work()