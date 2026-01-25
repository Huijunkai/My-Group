const { Sequelize } = require('sequelize');

const sequelize = new Sequelize('postgresql://postgres:eKtmtaNElgnVqHPBTQNIyyAjvLdcUBFR@switchyard.proxy.rlwy.net:51088/railway', {
    dialect: 'postgres',
    logging: false, // 设置为 console.log 可以查看 SQL 语句
    pool: {
        max: 5,
        min: 0,
        acquire: 30000,
        idle: 10000
    }
});

// 测试连接并同步模型
async function initDatabase() {
    try {
        await sequelize.authenticate();
        console.log('数据库连接成功');
        // sync({ alter: true }) 会根据模型定义自动更新表结构
        await sequelize.sync({ alter: true });
        console.log('所有模型已同步');
    } catch (error) {
        console.error('数据库连接或同步失败:', error);
    }
}

module.exports = {
    sequelize,
    initDatabase
};
