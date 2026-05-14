const { Sequelize } = require('sequelize');

const DB_NAME = process.env.DB_NAME || 'app_db';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASSWORD = process.env.DB_PASSWORD || '021219Hjk!';
const DB_HOST = process.env.DB_HOST || '127.0.0.1';
const DB_PORT = parseInt(process.env.DB_PORT) || 3306;

const sequelize = new Sequelize(
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    {
        host: DB_HOST,
        port: DB_PORT,
        dialect: 'mysql',
        logging: false,
        
        pool: {
            max: 20,
            min: 5,
            acquire: 30000,
            idle: 600000
        },
        
        define: {
            charset: 'utf8mb4',
            collate: 'utf8mb4_unicode_ci',
            underscored: false,
            freezeTableName: true
        },
        
        timezone: '+08:00'
    }
);

async function initDatabase() {
    try {
        await sequelize.authenticate();
        console.log('数据库连接成功 [华为云 MariaDB]');
        
        const models = require('./models');
        
        await sequelize.sync({ alter: true });
        console.log('所有模型已同步');
        return true;
    } catch (error) {
        console.error('数据库连接失败:', error.message);
        console.warn('将以无数据库模式启动，部分功能可能不可用');
        return false;
    }
}

// 先导出基本对象
module.exports = {
    sequelize,
    initDatabase
};

// 然后加载模型并添加到导出对象
const models = require('./models');
for (const key in models) {
    module.exports[key] = models[key];
}
