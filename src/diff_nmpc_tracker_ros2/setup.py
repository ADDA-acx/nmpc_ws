from setuptools import find_packages, setup
from glob import glob

package_name = 'diff_nmpc_tracker_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/diff_simple_tracking.launch.py']),
        ('share/' + package_name + '/launch', ['launch/diff_real_tracking.launch.py']),
        ('share/' + package_name + '/rviz',  # ← 新增
        glob('rviz/*.rviz')), 
    ],
    install_requires=[
        'setuptools',
        'acados_template',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='huangandong905@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'diff_model_node = diff_nmpc_tracker_ros2.diff_model_node:main',
            'diff_nmpc_node   = diff_nmpc_tracker_ros2.diff_nmpc_node:main',
            'ref_pub = diff_nmpc_tracker_ros2.ref_pub:main',
        ],
    },
)
