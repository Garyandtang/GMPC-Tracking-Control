clear;
gmpc_name = 'gmnc_new_05_5hz_100_00_1.3.bag';
nmpc_name = 'naive_new_01.bag';
max_num = 645;
font_size = 16;
line_width = 2;
t = linspace(0, 0.2*max_num, max_num);
vehicle_odoms_gmpc = get_traj(gmpc_name);
vehicle_odoms_nmpc = get_traj(nmpc_name);
vehicle_vels_gmpc = get_vel(gmpc_name);
vehicle_vels_nmpc = get_vel(nmpc_name);
circle = get_circle(2,0,2);

vehicle_oris_gmpc = get_orientation(gmpc_name);

roll = vehicle_oris_gmpc(:,2);
pitchs = vehicle_oris_gmpc(:,3);
labels = {'roll', 'pitch'};

font_size = 18;
line_width = 2;

h= figure;
set(h, 'DefaultTextFontSize', font_size);
boxHandles =boxplot([roll, pitchs], 'Labels', labels, 'Whisker', 1, 'Colors', 'b', 'symbol','');
% Set the fill color for each box
h_box=findobj(gca,'Tag','Box');
h_M = findobj(gca,'Tag','Median');

colors=['r','g','b','r','g','b'];
for j=1:length(h_box)
    h_M(j).Color='k';
    uistack(patch(get(h_box(j),'XData'),get(h_box(j),'YData'),colors(j),'FaceAlpha',1,'linewidth',1.5),'bottom');
end
 
set(gca,'color','none');
ylabel('degree (rad)');
set(gca,'XTick',1:2)%2 because only exists 2 boxplot
set(gca,'XTickLabel',labels,'FontSize',14)
% linear velocity

figure (1)
plot(t, vehicle_vels_gmpc(1:max_num,1), LineWidth=line_width);
hold on 
plot(t, vehicle_vels_nmpc(1:max_num,1),LineWidth=line_width);
% xlim([0, 0.2*max_num])
xlabel('t~(s)$', 'Interpreter','latex','FontSize',font_size)
ylabel('$v~(m/s)$', 'Interpreter','latex','FontSize',font_size)
legend("GMPC", 'NMPC', 'Interpreter','latex','FontSize',font_size-2)

% trajectory
figure (2)
plot(vehicle_odoms_gmpc(1:max_num,1), vehicle_odoms_gmpc(1:max_num,2), LineWidth=line_width);
hold on 
plot(vehicle_odoms_nmpc(1:max_num,1), vehicle_odoms_nmpc(1:max_num,2), LineWidth=line_width);
plot(circle(:,1), circle(:,2), LineWidth=1.5);
xlim([-2.1, 2.03])
ylim([-0.1, 4.1])
xlabel('x (m)', 'FontSize',font_size)
ylabel('y (m)', 'FontSize',font_size)
legend("GMPC", 'NMPC','reference', 'FontSize',font_size-2)

% angular velocity
figure (3)
plot(t, vehicle_vels_gmpc(1:max_num,2), LineWidth=line_width);
hold on 
plot(t, vehicle_vels_nmpc(1:max_num,2), LineWidth=line_width);
% xlim([0, 0.2*max_num])
xlabel('$t~(s)$', 'Interpreter','latex','FontSize',font_size)
ylabel('$w~(rad/s)$', 'Interpreter','latex','FontSize',font_size)
legend("GMPC", 'NMPC', 'Interpreter','latex','FontSize',font_size-2)

figure (4)
plot(vehicle_odoms_gmpc(1:max_num,1), vehicle_odoms_gmpc(1:max_num,2), LineWidth=line_width-0.5);
hold on 
plot(vehicle_odoms_nmpc(1:max_num,1), vehicle_odoms_nmpc(1:max_num,2), LineWidth=line_width-0.5);
plot(circle(:,1), circle(:,2), LineWidth=line_width-0.5);
% xlabel('$x~(m)$', 'Interpreter','latex','FontSize',font_size)
% ylabel('$y~(m)$', 'Interpreter','latex','FontSize',font_size)
xlim([1.85,  2.13])
ylim([2.08,2.26])


function vehicle_odoms = get_traj(bag_name)
bag = rosbag(bag_name);
bag_len = bag.NumMessages;
vehicle_odoms = zeros(bag_len,2);

current = select(bag,'Topic','/vel_odom','MessageType','nav_msgs/Odometry');
vehicle_current = readMessages(current,'DataFormat','struct');
for j=1:bag_len
    vehicle_odoms(j,1) = vehicle_current{j}.Pose.Pose.Position.X;
    vehicle_odoms(j,2) = vehicle_current{j}.Pose.Pose.Position.Y;
end
end

function vehicle_vels = get_vel(bag_name)
bag = rosbag(bag_name);
bag_len = bag.NumMessages;
vehicle_vels = zeros(bag_len,2);

current = select(bag,'Topic','/vel_odom','MessageType','nav_msgs/Odometry');
vehicle_current = readMessages(current,'DataFormat','struct');
for j=1:bag_len
    vehicle_vels(j,1) = vehicle_current{j}.Twist.Twist.Linear.X;
    vehicle_vels(j,2) = vehicle_current{j}.Twist.Twist.Angular.Z;
end
end

function vehicle_oris = get_orientation(bag_name)
bag = rosbag(bag_name);
bag_len = bag.NumMessages;
vehicle_oris = zeros(bag_len,3);

current = select(bag,'Topic','/vel_odom','MessageType','nav_msgs/Odometry');
vehicle_current = readMessages(current,'DataFormat','struct');
for j=1:bag_len
    w = vehicle_current{j}.Pose.Pose.Orientation.W;
    x = vehicle_current{j}.Pose.Pose.Orientation.X;
    y = vehicle_current{j}.Pose.Pose.Orientation.Y;
    z = vehicle_current{j}.Pose.Pose.Orientation.Z;
    eul = quat2eul([w x y z]);
    vehicle_oris(j,:) = eul;
end
end

function circle =  get_circle(radius, centerX, centerY)
    theta = linspace(0, 2*pi, 500); % 在0到2π之间生成100个角度
    x = radius * cos(theta) + centerX; % 计算 x 坐标
    y = radius * sin(theta) + centerY; % 计算 y 坐标

    circle = [x', y'];
end