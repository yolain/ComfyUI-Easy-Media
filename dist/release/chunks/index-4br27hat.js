var B=()=>{if(typeof crypto<"u"&&typeof crypto.randomUUID==="function")return crypto.randomUUID();let c=crypto.getRandomValues(new Uint8Array(16));c[6]=c[6]&15|64,c[8]=c[8]&63|128;let j=Array.from(c,(z)=>z.toString(16).padStart(2,"0")).join("");return`${j.slice(0,8)}-${j.slice(8,12)}-${j.slice(12,16)}-${j.slice(16,20)}-${j.slice(20)}`};
export{B as Sa};
