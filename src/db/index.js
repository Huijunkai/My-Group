const { Sequelize } = require('sequelize');

const sequelize = new Sequelize(
    'app_db',
    'app_backend',
    '021219Hjk!',
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
            timestamps: true,
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
        
        await sequelize.sync({ alter: true });
        console.log('所有模型已同步');
        
    } catch (error) {
        console.error('数据库连接失败:', error.message);
        process.exit(1);
    }
}

module.exports = { sequelize, initDatabase };
