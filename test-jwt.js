require('dotenv').config({ path: '/opt/jw-backend/jw-backend/.env' });
const fs = require('fs');
const crypto = require('crypto');

console.log('=== JWT 诊断脚本 ===\n');

// 1. 检查环境变量
console.log('1. 环境变量检查:');
console.log('   HUAWEI_KEY_FILE:', process.env.HUAWEI_KEY_FILE || 'NOT SET');
console.log('   HUAWEI_KEY_ID:', process.env.HUAWEI_KEY_ID ? 'SET (' + process.env.HUAWEI_KEY_ID.substring(0,10) + '...)' : 'NOT SET');
console.log('   HUAWEI_SUB_ACCOUNT:', process.env.HUAWEI_SUB_ACCOUNT || 'NOT SET');
console.log('   HUAWEI_PRIVATE_KEY:', process.env.HUAWEI_PRIVATE_KEY ? 'SET (length:' + process.env.HUAWEI_PRIVATE_KEY.length + ')' : 'NOT SET');

// 2. 尝试从文件加载
console.log('\n2. 文件加载检查:');
const keyFile = process.env.HUAWEI_KEY_FILE;
if (keyFile) {
    console.log('   文件路径:', keyFile);
    console.log('   文件存在:', fs.existsSync(keyFile));
    
    if (fs.existsSync(keyFile)) {
        try {
            const content = fs.readFileSync(keyFile, 'utf8');
            const config = JSON.parse(content);
            console.log('   JSON 解析: 成功');
            console.log('   key_id:', config.key_id);
            console.log('   sub_account:', config.sub_account);
            console.log('   private_key 长度:', config.private_key ? config.private_key.length : 0);
            console.log('   private_key 前50字符:', config.private_key ? config.private_key.substring(0, 50) : 'EMPTY');
            
            // 3. 测试 JWT 生成
            console.log('\n3. JWT 生成测试:');
            
            let privateKeyFormatted = config.private_key;
            if (!privateKeyFormatted.includes('-----BEGIN')) {
                privateKeyFormatted = `-----BEGIN PRIVATE KEY-----\n${privateKeyFormatted}\n-----END PRIVATE KEY-----`;
            }
            privateKeyFormatted = privateKeyFormatted.replace(/\\n/g, '\n');
            
            console.log('   格式化后私钥长度:', privateKeyFormatted.length);
            console.log('   格式化后私钥前30字符:', privateKeyFormatted.substring(0, 30));
            
            try {
                const header = { kid: config.key_id, typ: 'JWT', alg: 'PS256' };
                const payload = { aud: 'https://oauth-login.cloud.huawei.com/oauth2/v3/token', iss: config.sub_account, exp: Math.floor(Date.now()/1000) + 3600, iat: Math.floor(Date.now()/1000) };
                
                const encodedHeader = Buffer.from(JSON.stringify(header)).toString('base64').replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,'');
                const encodedPayload = Buffer.from(JSON.stringify(payload)).toString('base64').replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,'');
                const signingInput = `${encodedHeader}.${encodedPayload}`;
                
                const sign = crypto.createSign('RSA-SHA256');
                sign.update(signingInput);
                sign.end();
                
                const signature = sign.sign({
                    key: privateKeyFormatted,
                    padding: crypto.constants.RSA_PKCS1_PSS_PADDING,
                    saltLength: crypto.constants.RSA_PSS_SALT_LEN_DIGEST
                }, 'base64').replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,'');
                
                const jwt = `${signingInput}.${signature}`;
                console.log('   JWT 生成: 成功!');
                console.log('   JWT 长度:', jwt.length);
                console.log('   JWT 前50字符:', jwt.substring(0, 50));
            } catch (e) {
                console.log('   JWT 生成: 失败!');
                console.log('   错误信息:', e.message);
            }
        } catch (e) {
            console.log('   JSON 解析: 失败 -', e.message);
        }
    }
} else {
    console.log('   文件不存在!');
}

console.log('\n=== 诊断完成 ===');
