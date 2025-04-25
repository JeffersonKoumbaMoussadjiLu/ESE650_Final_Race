from setuptools import setup

package_name = ''

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kaRpuri',
    maintainer_email='kapuri@seas.upenn.edu',
    description='f1tenth pure_pursuit',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'final_race_node = final_race.final_race_node:main',
        ],
    },
)
