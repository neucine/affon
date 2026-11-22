#include <stdint.h>
#include <string.h>

#include "quickjs.h"

JSValue affon_js_new_number_bits(JSContext *context, long long bits) {
    double value;
    memcpy(&value, &bits, sizeof(value));
    return JS_NewNumber(context, value);
}
