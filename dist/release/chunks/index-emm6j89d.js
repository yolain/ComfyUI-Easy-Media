var g=()=>{if(typeof crypto<"u"&&typeof crypto.randomUUID==="function")return crypto.randomUUID();return`${Date.now().toString(36)}-${Math.random().toString(36).slice(2,11)}`};
export{g as ha};
