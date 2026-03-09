// const { Sequelize } = require('sequelize');

// // 优先从环境变量读取数据库连接串，便于在 Railway 等平台上切换实例：
// const dbUrl =
//     process.env.DATABASE_URL ||
//     process.env.MYSQL_URL ||
//     process.env.MYSQLDATABASE_URL ||
//     'mysql://root:ZsanAhqaOCNMgsDccbrhEZrnBOFmVOcK@centerbeam.proxy.rlwy.net:21332/railway';

// const sequelize = new Sequelize(dbUrl, {
//     dialect: 'mysql',
//     logging: false, // 设置为 console.log 可以查看 SQL 语句
//     pool: {
//         max: 5,
//         min: 0,
//         acquire: 30000,
//         idle: 10000
//     },
//     // MySQL 特定配置
//     define: {
//         charset: 'utf8mb4',
//         collate: 'utf8mb4_unicode_ci'
//     }
// });

// // 测试连接并同步模型
// async function initDatabase() {
//     try {
//         await sequelize.authenticate();
//         console.log('数据库连接成功');
//         // sync({ alter: true }) 会根据模型定义自动更新表结构
//         // 如果表不存在会自动创建
//         await sequelize.sync({ alter: true });
//         console.log('所有模型已同步');
//     } catch (error) {
//         console.error('数据库连接或同步失败:', error);
//     }
// }

// module.exports = {
//     sequelize,
//     initDatabase
// };




const { Sequelize } = require('sequelize');

// ============================================
// 华为云 MariaDB 配置（同一服务器使用 localhost）
// ============================================

const sequelize = new Sequelize(
    'app_db',           // 数据库名
    'app_backend',      // 用户名
    '021219Hjk!',       // 密码
    {
        host: '127.0.0.1',      // ✅ 改为本地回环（不是192.168.0.237）
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
            underscored: true
        },
        
        timezone: '+08:00'
    }
);

// 测试连接
async function initDatabase() {
    try {
        await sequelize.authenticate();
        console.log('✅ 数据库连接成功 [本地 MariaDB]');
        
        await sequelize.sync({ 
            alter: process.env.NODE_ENV === 'development' 
        });
        console.log('✅ 所有模型已同步');
        
    } catch (error) {
        console.error('❌ 数据库连接失败:', error.message);
        process.exit(1);
    }
}

module.exports = { sequelize, initDatabase };
