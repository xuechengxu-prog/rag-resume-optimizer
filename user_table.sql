-- 创建 user 表
CREATE TABLE user (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
    name VARCHAR(50) NOT NULL COMMENT '姓名',
    age INT COMMENT '年龄',
    sex VARCHAR(10) COMMENT '性别',
    username VARCHAR(50) NOT NULL UNIQUE COMMENT '用户名',
    password VARCHAR(255) NOT NULL COMMENT '密码'
) COMMENT='用户表';

-- 插入示例数据
INSERT INTO user (id, name, age, sex, username, password) VALUES
(1, '小红', 22, '女', 'xiaohong', '123');
