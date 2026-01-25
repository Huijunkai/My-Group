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

        // 仅对 Course(课程表) 做结构收敛：移除不再需要的拆分字段列
        // 说明：sequelize.sync({ alter: true }) 在部分情况下不会自动 drop 列，这里显式处理一次
        try {
            const qi = sequelize.getQueryInterface();
            const columns = await qi.describeTable('Courses');
            const deprecatedColumns = ['startWeek', 'endWeek', 'isOdd', 'isEven', 'startPeriod', 'endPeriod'];
            for (const col of deprecatedColumns) {
                if (columns[col]) {
                    await qi.removeColumn('Courses', col);
                }
            }
            console.log('Course 表结构已收敛');
        } catch (e) {
            console.warn('Course 表结构收敛失败（可忽略）:', e.message);
        }
    } catch (error) {
        console.error('数据库连接或同步失败:', error);
    }
}

module.exports = {
    sequelize,
    initDatabase
};
