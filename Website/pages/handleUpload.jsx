import axios from "axios";
import {useState} from 'react'

export default function handleUpload(){


    axios({
        url:"truthcamera.tech/api/upload",
        method: 'POST',
        Headers:{
        },
        data: formdata
    }).then((res)=>{
        // handle reply
    }).catch(err=>console.log(err))
}