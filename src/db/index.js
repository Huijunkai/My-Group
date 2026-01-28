const { Sequelize } = require('sequelize');

// 使用 mysql2 驱动连接 MySQL 数据库
// 注意：url 中的协议头 mysql:// 会自动识别为 mysql 此时需要安装 mysql2
const sequelize = new Sequelize('mysql://root:qhJgaOeqFadSycseeWDiBIUZzQFyIHsm@yamanote.proxy.rlwy.net:13428/railway', {
    dialect: 'mysql',
    logging: false, // 设置为 console.log 可以查看 SQL 语句
    pool: {
        max: 5,
        min: 0,
        acquire: 30000,
        idle: 10000
    },
    // MySQL 特定配置
    dialectOptions: {
        // 如果需要处理日期时间时区问题，可以在这里配置
        dateStrings: true,
        typeCast: true
    },
    define: {
        // 全局配置：使用反引号作为标识符引用（MySQL 默认）
        // freezeTableName: false,
    }
});

// 测试连接并同步模型
async function initDatabase() {
    try {
        await sequelize.authenticate();
        console.log('数据库连接成功');
        
        // sync({ alter: true }) 会根据模型定义自动更新表结构
        // 对于新数据库，这会自动创建所有表和正确的索引/主键
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
