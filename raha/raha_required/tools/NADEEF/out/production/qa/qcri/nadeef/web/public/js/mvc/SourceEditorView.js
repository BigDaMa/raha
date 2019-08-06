/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

define([
    'state',
    "mvc/SourceEditorView",
    'text!mvc/template/sourceeditor.template.html'
], function (State, SourceEditor, SourceEditorTemplate) {
    "use strict";
    function start(id) {
        var html =  _.template(SourceEditorTemplate)();
        $('#' + id).html(html);
        var schema = "";
        // initialize the filedrop
        $("#dropbox").filedrop({
            maxfiles: 1,
            maxfilesize: 500,
            queuefiles: 1,
            url: "/do/upload",
            allowedfileextensions: ['.csv'],
            data: function () {
                return {
                    project: State.get("project"),
                    schema : schema
                };
            },
            error: function (err, file) {
                switch (err) {
                    case 'BrowserNotSupported':
                        alert('Your browser does not support HTML5 file uploads!');
                        break;
                    case 'TooManyFiles':
                        alert('Too many files! Please select 5 at most!');
                        break;
                    case 'FileTooLarge':
                        alert(file.name + ' is too large! Please upload files up to 500 mb.');
                        break;
                    default:
                        break;
                }
            },

            beforeSend: function (file, fileIndex, e, done) {
                $("#schema").removeClass("hide");
                var content = e.target.result;
                var i = 0;
                for (i = 0; i < content.length; i ++) {
                    if (content[i] === '\n') {
                        break;
                    }
                }

                var header = content.slice(0, i);
                var tbody = $("#schema").find("tbody");
                var tokens = header.split(",");
                var template =
                    '<tr>' +
                    '<td><input type="text" value="<%= value %>"></td>' +
                    '<td><select>' +
                    '<option value="string">String</option>' +
                    '<option value="integer">Integer</option>' +
                    '<option value="float">Float</option>' +
                    '</select></td></tr>';

                var html = '';
                for (i = 0; i < tokens.length; i ++) {
                    tokens[i] = tokens[i].trim();
                    html += _.template(template, { value : tokens[i] });
                }

                tbody.empty().append(html);
                $("#source-editor-save").on("click", function (e) {
                    var attributes = $("#schema").find("tbody input");
                    var types = $("#schema").find("tbody select");
                    for (var i = 0; i < attributes.length; i ++) {
                        var token = "";
                        if (i != 0)
                            token = ",";
                        // TODO: needs to process column name to remove quote or space
                        token += attributes[i].value.trim() + " ";
                        if (types[i].value.trim() === "integer")
                            token += "int";
                        else if (types[i].value.trim() === "float")
                            token += "float";
                        else
                            token += "string";
                        schema += token;
                    }

                    done();
                });

                $("#source-editor-save").removeClass("disabled");
            },

            uploadStarted: function (i, file, len) {
                // a file began uploading
                // i = index => 0, 1, 2, 3, 4 etc
                // file is the actual file of the index
                // len = total files user dropped
                $("#dropbox-message").text('Uploading ' + file.name);
                $("#source-editor-save").prop('disabled', true);
                $("#source-editor-cancel").prop('disabled', true);
            },

            uploadFinished: function (i, file, response, time) {
                // response is the data you got back from server in JSON format.
                $("#dropbox-message").text('Drop CSV file here to upload.');
                $("#source-editor-save").prop('disabled', false);
                $("#source-editor-cancel").prop('disabled', false);
                $('#dropbox-progress div').width(0);
                $('#schema').addClass('hide');
                $("#source-editor-save").addClass("disabled");
            },

            progressUpdated: function (i, file, progress) {
                // this function is used for large files and updates intermittently
                // progress is the integer value of file being uploaded percentage to completion
            },

            globalProgressUpdated: function (progress) {
                // progress for all the files uploaded on the current instance (percentage)
                // ex: $('#progress div').width(progress+"%");
                $('#dropbox-progress div').width(progress + "%");
            }
        });

        $("#source-editor").on("hidden", function () {
            $("#controller")[0].dispatchEvent(new Event("refreshSource"));
        });
    }

    return {
        start: start
    };
});
