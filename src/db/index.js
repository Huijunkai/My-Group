const { Sequelize } = require('sequelize');

// 优先从环境变量读取数据库连接串，便于在 Railway 等平台上切换实例：
// 例如：mysql://user:password@nozomi.proxy.rlwy.net:24647/railway
const dbUrl =
    process.env.DATABASE_URL ||
    process.env.MYSQL_URL ||
    process.env.MYSQLDATABASE_URL ||
    'mysql://root:qhJgaOeqFadSycseeWDiBIUZzQFyIHsm@yamanote.proxy.rlwy.net:13428/railway';

const sequelize = new Sequelize(dbUrl, {
    dialect: 'mysql',
    logging: false, // 设置为 console.log 可以查看 SQL 语句
    pool: {
        max: 5,
        min: 0,
        acquire: 30000,
        idle: 10000
    },
    // MySQL 特定配置
    define: {
        charset: 'utf8mb4',
        collate: 'utf8mb4_unicode_ci'
    }
});

// 测试连接并同步模型
async function initDatabase() {
    try {
        await sequelize.authenticate();
        console.log('数据库连接成功');
        // sync({ alter: true }) 会根据模型定义自动更新表结构
        // 如果表不存在会自动创建
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
