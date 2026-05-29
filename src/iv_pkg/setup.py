from setuptools import find_packages, setup

package_name = 'iv_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mae00018',
    maintainer_email='mae00018@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],  # Modern replacement for tests_require
    },
    entry_points={
        'console_scripts': [
        	'pure_pursuit_train = iv_pkg.pure_pursuit_train:main',
        	'pure_pursuit_eval = iv_pkg.pure_pursuit_eval:main',
        	'train_rl = iv_pkg.train_rl:main',
        	'waypoint = iv_pkg.waypoint:main',
        	'pure_pursuit= iv_pkg.pure_pursuit:main',
        	'fixed= iv_pkg.fixed:main',
        ],
    },
)
