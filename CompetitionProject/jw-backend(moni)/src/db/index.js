const { Sequelize } = require('sequelize');

const sequelize = new Sequelize(
    'app_db',
    'root',
    '',
    {
        host: '127.0.0.1',
        port: 3306,
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
