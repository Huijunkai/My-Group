const axios = require('axios');

async function test() {
    try {
        const response = await axios.post('http://localhost:3000/api/water/scan', {
            scanUrl: 'http://wx.happy-ti.com/wxpay/scanqrcode/v0.html?openid=oojMD2E44dYFhoycLuov5o1lRldw&deviceid=861290071439769&app=WECHAT&token=&ch='
        });
        console.log('Response:', JSON.stringify(response.data, null, 2));
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

test();
