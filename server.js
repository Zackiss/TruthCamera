const express = require('express');
const app = express();
const cors = require('cors'); // what the fuck is this?

app.use(cors())

app.post("/api/upload",(req,res)=>{

    console.log(req.body);
    console.log('file uploaded');
    setTimeout(()=>{
        console.log('file uploaded')
        return res.status(200).json({result:false})
    },3000);    
})


app.listen(8080,()=>{
    console.log("Server running at port 8080");
})