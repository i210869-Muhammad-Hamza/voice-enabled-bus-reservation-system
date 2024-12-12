-- Create the database
CREATE DATABASE `bus_reservation_system`;

-- Use the created database
USE `bus_reservation_system`;

-- Create the 'user' table
CREATE TABLE `user` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(100) NOT NULL,
    `password` VARCHAR(255) NOT NULL
);

-- Create the 'bus' table
CREATE TABLE `bus` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `capacity` INT NOT NULL
);

-- Create the 'schedule' table
CREATE TABLE `schedule` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `start_city` VARCHAR(100) NOT NULL,
    `end_city` VARCHAR(100) NOT NULL,
    `bus_id` INT NOT NULL,
    `fare` DECIMAL(10, 2) NOT NULL,
    `departure_date` DATE NOT NULL,
    `departure_time` TIME NOT NULL,
    FOREIGN KEY (`bus_id`) REFERENCES `bus`(`id`)
);

-- Insert sample data into 'user' table
INSERT INTO `user` (`name`, `password`) VALUES
('Ali Ahmed', 'password123'),
('Fatima Khan', 'securepass'),
('Hamza Ali', 'mypassword'),
('Sara Noor', 'abc123');

-- Insert sample data into 'bus' table
INSERT INTO `bus` (`capacity`) VALUES
(40), 
(45), 
(50), 
(30);

-- Insert sample data into 'schedule' table
INSERT INTO `schedule` (`start_city`, `end_city`, `bus_id`, `fare`, `departure_date`, `departure_time`) VALUES
('Karachi', 'Lahore', 1, 3500.00, '2024-11-30', '08:00:00'),
('Islamabad', 'Multan', 2, 1500.00, '2024-12-01', '10:30:00'),
('Peshawar', 'Quetta', 3, 4500.00, '2024-12-02', '09:00:00'),
('Faisalabad', 'Rawalpindi', 4, 1200.00, '2024-12-03', '07:45:00'),
('Sialkot', 'Hyderabad', 1, 4000.00, '2024-12-04', '06:00:00');

