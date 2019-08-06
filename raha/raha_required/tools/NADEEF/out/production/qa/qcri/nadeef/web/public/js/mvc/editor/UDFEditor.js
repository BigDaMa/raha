define([
    "text!mvc/editor/template/udf.template.html"
], function (Template) {
    "use strict";
    function UDFEditor(dom, table1, table2, rule) {
        this.dom = dom;
        this.table1 = table1;
        this.table2 = table2;
        this.rule = rule;
    }

    UDFEditor.prototype.render = function () {
        $(this.dom).html(_.template(Template));

    };

    UDFEditor.prototype.val = function () {
        return null;
    };

    return {
        Create: UDFEditor
    };
});