from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.cli import CLI
import pandas as pd

def inject_traffic(dataset):
    # Simulate SQL injection attack targeting a specific host
    # For example, let's target the FIT101 sensor with an SQL injection attack
    attack_host = 'FIT101'
    attack_command = "'; DROP TABLE SensorData; --"

    # Modify the dataset to inject the SQL injection command
    dataset.loc[dataset['Timestamp'] == attack_host, 'FIT101'] = attack_command

    # You can perform other actions like sending packets, etc.
    print(f"SQL injection attack sent to host {attack_host}")

def create_topology(dataset):
    net = Mininet(controller=None, switch=OVSSwitch)

    # Add switches
    s1 = net.addSwitch('s1')

    # Add hosts (representing sensors and actuators)
    sensors = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'FIT301', 'LIT301', 'AIT401', 'AIT402']
    actuators = ['MV101', 'P101', 'P102', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404']
    for sensor in sensors:
        net.addHost(sensor)
    for actuator in actuators:
        net.addHost(actuator)

    # Connect hosts to switches
    for sensor in sensors:
        net.addLink(sensor, s1)
    for actuator in actuators:
        net.addLink(actuator, s1)

    # Start Mininet
    net.start()

    # Start the controller (POX)
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
    c0.start()

    # Assign controller to the switch
    s1.start([c0])

    # Inject traffic based on the dataset
    inject_traffic(dataset)

    # Start the CLI
    CLI(net)

    # Stop Mininet
    net.stop()

if __name__ == '__main__':
    # Read the CSV file skipping the first row
    dataset = pd.read_csv('SWaT_Dataset_Normal_v1_modified.csv', skiprows=1)

    # Write the modified dataset to a new CSV file
    # dataset.to_csv('SWaT_Dataset_Normal_v1_modified.csv', index=False)
    dataset.columns = ['Timestamp', 'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203',
                       'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301',
                       'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401',
                       'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
                       'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503',
                       'FIT601', 'P601', 'P602', 'P603', 'Normal/Attack']
    # Create topology and inject traffic
    create_topology(dataset)

