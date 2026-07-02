%%
%% This is file `kulemt.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% kulemt-start.dtx  (with options: `initial')
%% kulemt-version.dtx  (with options: `fileinfo')
%% kulemt-start.dtx  (with options: `class')
%% kulemt-cfg.dtx  (with options: `class')
%% kulemt-opt.dtx  (with options: `class')
%% kulemt-load.dtx  (with options: `class')
%% kulemt-layout.dtx  (with options: `class')
%% kulemt-front.dtx  (with options: `class')
%% kulemt-extra.dtx  (with options: `class')
%% kulemt-load.dtx  (with options: `atend')
%% 
%% Copyright (C) 2025 by Luc Van Eycken
%% 
%% It may be distributed and/or modified under the conditions of
%% the LaTeX Project Public License (LPPL), either version 1.3c of
%% this license or (at your option) any later version.  The latest
%% version of this license is in the file:
%% 
%%    https://www.latex-project.org/lppl.txt
%% 
\NeedsTeXFormat{LaTeX2e}[2018/12/01]
\DeclareRelease{v1}{2010-03-02}{kulemt-v1.cls}
\DeclareCurrentRelease{}{2025-01-01}
\@ifundefined{ExplFileDate}{\RequirePackage{expl3}}{}
\@ifl@t@r\ExplFileDate{2019-04-06}{}{%
  \ClassError{kulemt}%
    {The L3 programming layer is too old}%
    {You need to update your installation of 'l3kernel'.\MessageBreak
     Loading 'kulemt' will abort.}%
  \endinput}
\@ifundefined{NewDocumentCommand}{\RequirePackage{xparse}}{}
\def\filename{kulemt.cls}
\def\filedate{2025-05-20}
\def\fileversion{2.1.0}
\edef\fileinfo{KU\space Leuven\space Engineering\space Master's\space
  Thesis\space document\space class}
\ProvidesExplClass{kulemt}{\filedate}{\fileversion}{\fileinfo}
\cs_if_free:NT \legacy_if:nTF
  {
    \prg_new_conditional:Npnn \legacy_if:n #1 { TF }
      {
        \exp_after:wN \reverse_if:N
          \cs:w if#1 \cs_end:
          \prg_return_false:
        \else:
          \prg_return_true:
        \fi:
      }
  }
\cs_if_free:NT \sys_if_engine_opentype:TF
  {
    \cs_if_exist:NTF \Umathcode
      {
        \cs_new_eq:NN \sys_if_engine_opentype:T  \use:n
        \cs_new_eq:NN \sys_if_engine_opentype:F  \use_none:n
        \cs_new_eq:NN \sys_if_engine_opentype:TF \use_i:nn
      }
      {
        \cs_new_eq:NN \sys_if_engine_opentype:T  \use_none:n
        \cs_new_eq:NN \sys_if_engine_opentype:F  \use:n
        \cs_new_eq:NN \sys_if_engine_opentype:TF \use_ii:nn
      }
  }
\cs_new:Nn \kulemt_keys_key:
  {
    \cs_if_exist_use:NF \l_keys_key_str
      { \tl_to_str:V \l_keys_key_tl }
  }
\cs_if_exist:NF \IfFormatAtLeastTF
  { \cs_new:Npn \IfFormatAtLeastTF { \@ifl@t@r \fmtversion } }
\cs_if_exist:NF \IfPackageLoadedTF
  { \cs_set_eq:NN \IfPackageLoadedTF \@ifpackageloaded }
\cs_if_exist:NF \ProcessKeyOptions
  {
    \RequirePackage {l3keys2e}
    \NewDocumentCommand \ProcessKeyOptions { O{\@currname} }
      { \exp_args:No \ProcessKeysOptions {#1} }
  }
\int_compare:nNnT { \tracinglostchars } < {2}
  { \int_set:Nn \tracinglostchars {2} }
\IfFormatAtLeastTF {2020-10-01}
  {
    \cs_new_protected:Nn \kulemt_at_end_preamble:n
      { \AddToHook {begindocument/before} {#1} }
  }
  {
    \tl_new:N \l__kulemt_begindocument_before_tl
    \cs_new_protected:Nn \kulemt_at_end_preamble:n
      { \tl_put_right:Nn \l__kulemt_begindocument_before_tl {#1} }
    \tl_put_left:Nn \document
      {
        \group_end:
        \l__kulemt_begindocument_before_tl
        \group_begin:
      }
  }
\IfFormatAtLeastTF {2020-10-01}
  {
    \cs_new_protected:Nn \kulemt_after_begin_document:n
      { \AddToHook {begindocument/end} {#1} }
  }
  {
    \tl_new:N \l__kulemt_begindocument_end_tl
    \cs_new_protected:Nn \kulemt_after_begin_document:n
      { \tl_put_right:Nn \l__kulemt_begindocument_end_tl {#1} }
    \tl_put_right:Nn \document
      {
        \l__kulemt_begindocument_end_tl
        \ignorespaces
      }
  }
\cs_generate_variant:Nn \exp_last_unbraced:Nn { Ne }
\cs_generate_variant:Nn \msg_error:nnn { nne }
\cs_generate_variant:Nn \msg_error:nnnn { nnee }
\cs_generate_variant:Nn \msg_error:nnnnn { nneee }
\cs_generate_variant:Nn \msg_fatal:nnn { nne }
\cs_generate_variant:Nn \msg_warning:nnnn { nnee }
\cs_generate_variant:Nn \prop_put:Nnn { Nne }
\cs_generate_variant:Nn \str_put_right:Nn { Ne }
\cs_generate_variant:Nn \str_set:Nn { Ne }
\cs_generate_variant:Nn \tl_set:Nn { Ne }
\seq_new:N \l_kulemt_masters_seq
\prop_new:N \l_kulemt_master_prop
\prop_new:N \l_kulemt_cfg_prop
\prop_set_from_keyval:Nn \l_kulemt_cfg_prop
  {
    date = ,
    text.and.dutch   = en ,
    text.and.english = and ,
    text.assessor.dutch   = evaluator ,
    text.assessor.english = assessor ,
    text.assessor.plural.dutch   = evaluatoren ,
    text.assessor.plural.english = assessors ,
    text.assistant.dutch   = begeleider ,
    text.assistant.english = assistant-supervisor ,
    text.assistant.plural.dutch   = begeleiders ,
    text.assistant.plural.english = assistant-supervisors ,
    text.author.dutch   = auteur ,
    text.author.english = author ,
    text.author.plural.dutch   = auteurs ,
    text.author.plural.english = authors ,
    text.copyright.dutch   = { Alle~ rechten~ voorbehouden.~ Niets~ uit~ deze~
      uitgave~ mag~ worden~ vermenigvuldigd~ en/of~ openbaar~ gemaakt~ worden~
      door~ middel~ van~ druk,~ fotokopie,~ microfilm,~ elektronisch~ of~ op~
      welke~ andere~ wijze~ ook~ zonder~ voorafgaande~ schriftelijke~
      toestemming~ van~ de~ uitgever. } ,
    text.copyright.english = { All~ rights~ reserved.~ No~ part~ of~ the~
      publication~ may~ be~ reproduced~ in~ any~ form~ by~ print,~ photoprint,~
      microfilm,~ electronic~ or~ any~ other~ means~ without~ written~
      permission~ from~ the~ publisher. } ,
    text.acyear.pre.dutch   = academiejaar ,
    text.acyear.pre.english = academic~ year ,
    text.title.pre.dutch   = thesis~ voorgedragen~ tot~ het~ behalen~
                             van~ de~ graad~ van ,
    text.title.pre.english = thesis~ submitted~ for~ the~ degree~ of ,
    text.promoter.dutch   = promotor ,
    text.promoter.english = supervisor ,
    text.promoter.plural.dutch   = promotoren ,
    text.promoter.plural.english = supervisors ,
    text.publisher.pre.dutch   = uitgegeven~ in~ eigen~ beheer~ door ,
    text.publisher.pre.english = published~ by
  }
\prop_map_inline:Nn \l_kulemt_cfg_prop
  { \prop_put:Nne \l_kulemt_cfg_prop {#1} { \tl_to_str:n {#2} } }
\cs_generate_variant:Nn \prop_put:Nnn { Ne }
\cs_generate_variant:Nn \seq_set_split:Nnn { Nee }
\prg_generate_conditional_variant:Nnn \seq_if_in:Nn { Ne } { T, F, TF }
\prg_generate_conditional_variant:Nnn \str_case:nn { V } { T, F, TF }
\cs_new_protected:Nn \__kulemt_cfg_str_seq_const_from_clist:Nn
  { \exp_args:NNe \seq_const_from_clist:Nn #1 { \tl_to_str:n {#2} } }
\cs_new_protected:Nn \__kulemt_cfg_str_with_utf:N
  {
    \tl_set_rescan:Nno #1
      {
        \ExplSyntaxOff
        \seq_map_inline:Nn \l_char_special_seq
          { \char_set_catcode_other:N ##1 }
        \char_set_catcode_space:n {9}
        \char_set_catcode_space:n {32}
      }
      { #1 }
  }
\cs_new_protected:Nn \__kulemt_cfg_str_with_utf:nN
  {
    \seq_if_in:NnF \l__kulemt_cfg_non_str_items_seq {#1}
      { \__kulemt_cfg_str_with_utf:N #2 }
  }
\seq_const_from_clist:Nn \l__kulemt_cfg_non_str_items_seq { options }
\sys_if_engine_opentype:T
  {
    \cs_set_eq:NN \__kulemt_cfg_str_with_utf:N \use_none:n
    \cs_set_eq:NN \__kulemt_cfg_str_with_utf:nN \use_none:nn
  }
\str_new:N  \l__kulemt_cfg_current_section_name_str
\str_new:N  \l__kulemt_cfg_current_key_str
\str_new:N  \l__kulemt_cfg_current_value_str
\prop_new:N \l__kulemt_cfg_current_section_prop
\seq_new:N \l__kulemt_cfg_current_options_seq
\prop_new:N \l__kulemt_cfg_section_defaults_prop
\__kulemt_cfg_str_seq_const_from_clist:Nn \c__kulemt_cfg_allowed_key_seq
  { name, type, language, option,
    faculty, faculty.dutch, faculty.english,
    faculty.logo, faculty.logo.dutch, faculty.logo.english,
    contact.address, contact.address.english, contact.address.dutch,
    contact.email, contact.phone }
\__kulemt_cfg_str_seq_const_from_clist:Nn \c__kulemt_cfg_language_values_seq
  { english, dutch }
\__kulemt_cfg_str_seq_const_from_clist:Nn \c__kulemt_cfg_option_values_seq
  { allowed, forbidden, required }
\__kulemt_cfg_str_seq_const_from_clist:Nn \c__kulemt_cfg_type_values_seq
  { initial, advanced }
\tl_new:N \l__kulemt_cfg_tmp_tl
\cs_new_protected:Nn \kulemt_read_config_file:
  {
    \prop_clear:N \l__kulemt_cfg_section_defaults_prop
    \seq_clear:N \l_kulemt_masters_seq
    \exp_args:NV \__kulemt_cfg_read_file:nn \l_kulemt_opt_cfgfile_tl {}
    \seq_if_empty:NT \l_kulemt_masters_seq
      { \msg_fatal:nne {kulemt} {cfg/no-masters} { \l_kulemt_opt_cfgfile_tl } }
  }
\msg_new:nnnn {kulemt} {cfg/no-masters}
  { No~ master~ definitions~ are~ found~ in~ the~ configuration~ file~ '#1'. }
  { Please~ correct~ the~ configuration~ file~ or~ select~ another~ one. }
\cs_new_protected:Nn \kulemt_read_config_file:n
  {
    \seq_if_in:NeF \l_kulemt_masters_seq { \tl_to_str:n {#1} }
      {
        \exp_args:NV \__kulemt_cfg_read_file:nn \l_kulemt_opt_cfgfile_tl {#1}
        \seq_if_in:NeF \l_kulemt_masters_seq { \tl_to_str:n {#1} }
          { \msg_fatal:nnn {kulemt} {cfg/missing-master} {#1} }
      }
  }
\msg_new:nnnn {kulemt} {cfg/missing-master}
  {
    The~ master~ abbreviation~ '#1'~ is~ not~ defined~
    in~ the~ configuration~ file~ '\l_kulemt_opt_cfgfile_tl'.
  }
  {
    Perhaps~ you~ mistyped~ the~ abbrevation~ or~
    the~ configuration~ file~ needs~ updating.
  }
\cs_new_protected:Nn \__kulemt_cfg_read_file:nn
  {
    \file_if_exist:nF {#1}
      { \msg_fatal:nnn {kulemt} {cfg/file-not-found} {#1} }
    \str_clear:N \l__kulemt_cfg_current_section_name_str
    \ior_open:Nn \g_tmpa_ior {#1}
    \ior_str_map_inline:Nn \g_tmpa_ior
      {
        \str_set:Ne \l_tmpa_str { \str_head_ignore_spaces:n {##1} }
        \str_case:VnF \l_tmpa_str
          {
            {}  {}
            {;} {}
            {[} {
                  \__kulemt_cfg_finish_read_section:
                  \tl_if_empty:nTF {#2}
                    { \__kulemt_cfg_start_read_section:nn {##1} {#2} }
                    {
                      \str_if_eq:VnTF
                        \l__kulemt_cfg_current_section_name_str {#2}
                        { \ior_map_break: }
                        { \__kulemt_cfg_start_read_section:nn {##1} {#2} }
                    }
                }
          }
          { \__kulemt_cfg_read_key:w ##1 == \q_stop }
      }
    \__kulemt_cfg_finish_read_section:
    \ior_close:N \g_tmpa_ior
  }
\msg_new:nnn {kulemt} {cfg/file-not-found}
  { The~ configuration~ file~ '#1'~ is~ missing. }
\cs_new_protected:Nn \__kulemt_cfg_start_read_section:nn
  {
    \exp_last_unbraced:Nf \__kulemt_cfg_start_read_section_aux:w
      { \str_tail_ignore_spaces:n {#1} } ] \q_stop
    \tl_if_empty:nT {#2}
      {
        \seq_if_in:NVT \l_kulemt_masters_seq
          \l__kulemt_cfg_current_section_name_str
          {
            \msg_error:nne {kulemt} {cfg/section-redefined}
              { \l__kulemt_cfg_current_section_name_str }
          }
      }
    \prop_clear:N \l__kulemt_cfg_current_section_prop
    \str_clear:N \l__kulemt_cfg_current_key_str
    \str_clear:N \l__kulemt_cfg_current_value_str
    \seq_clear:N \l__kulemt_cfg_current_options_seq
  }
\cs_new_protected:Npn \__kulemt_cfg_start_read_section_aux:w #1] #2 \q_stop
  {
    \tl_if_empty:nT {#2}
      { \msg_warning:nnn {kulemt} {cfg/section-name-error} {#1} }
    \str_set:Nn \l__kulemt_cfg_current_section_name_str {#1}
  }
\msg_new:nnnn {kulemt} {cfg/section-redefined}
  { Section~ '#1'~ defined~ more~ than~ once~ in~ the~ configuration~ file. }
  { The~ previous~ definition~ will~ be~ ignored. }
\msg_new:nnnn {kulemt} {cfg/section-name-error}
  { '[#1'~ misses~ a~ trailing~ ']'~ in~ the~ configuration~ file. }
  { An~ extra~ ']'~ is~ assumed. }
\cs_new_protected:Nn \__kulemt_cfg_finish_read_section:
  {
    \__kulemt_cfg_finish_read_key:
    \str_if_eq:VnTF \l__kulemt_cfg_current_section_name_str {defaults}
      {
        \prop_map_tokens:Nn \l__kulemt_cfg_current_section_prop
          { \prop_put:Nnn \l__kulemt_cfg_section_defaults_prop }
      }
      {
        \prop_clear:N \l_tmpa_prop
        \prop_put:NnV \l_tmpa_prop {} \l__kulemt_cfg_current_options_seq
        \prop_put:NnV \l__kulemt_cfg_current_section_prop {options} \l_tmpa_prop
        \str_if_empty:NF \l__kulemt_cfg_current_section_name_str
          { \seq_put_right:NV \l_kulemt_masters_seq
              \l__kulemt_cfg_current_section_name_str }
        \prop_set_eq:cN
          { l__kulemt_cfg_section_ \l__kulemt_cfg_current_section_name_str _prop }
          \l__kulemt_cfg_current_section_prop
        \prop_clear:c
          { l__kulemt_cfg_master_ \l__kulemt_cfg_current_section_name_str _prop }
      }
  }
\cs_new_protected:Npn \__kulemt_cfg_read_key:w #1 = #2 = #3 \q_stop
  {
    \str_case:nnF {#3}
      {
        {}  {
              \str_put_right:Ne \l__kulemt_cfg_current_value_str
                { \c_space_tl \tl_trim_spaces:n {#1} }
            }
        {=} {
              \__kulemt_cfg_finish_read_key:
              \str_set:Ne \l__kulemt_cfg_current_key_str
                { \tl_trim_spaces:n {#1} }
              \str_set:Ne \l__kulemt_cfg_current_value_str
                { \tl_trim_spaces:n {#2} }
            }
      }
      {
        \__kulemt_cfg_finish_read_key:
        \str_set:Ne \l__kulemt_cfg_current_key_str { \tl_trim_spaces:n {#1} }
        \str_set:Ne \l__kulemt_cfg_current_value_str
          { \tl_trim_spaces:o { \__kulemt_cfg_read_key_aux:w #2 = #3 \q_stop } }
      }
  }
\cs_new:Npn \__kulemt_cfg_read_key_aux:w #1 == \q_stop { #1 }
\cs_new_protected:Nn \__kulemt_cfg_finish_read_key:
  {
    \str_if_empty:NF \l__kulemt_cfg_current_key_str
      {
        \str_if_empty:NTF \l__kulemt_cfg_current_section_name_str
          {
            \prop_put:NVV \l_kulemt_cfg_prop \l__kulemt_cfg_current_key_str
              \l__kulemt_cfg_current_value_str
          }
          {
            \seq_set_split:NnV \l_tmpa_seq {.} \l__kulemt_cfg_current_key_str
            \seq_pop_left:NN \l_tmpa_seq \l_tmpa_str
            \str_set:Ne \l_tmpb_str { \seq_use:Nn \l_tmpa_seq {.} }
            \str_if_eq:VnTF \l_tmpb_str {from}
              {
                \__kulemt_cfg_get_section_item_from:VVN
                  \l__kulemt_cfg_current_value_str \l_tmpa_str \l_tmpa_prop
              }
              {
                \seq_if_in:NVTF \c__kulemt_cfg_allowed_key_seq
                  \l__kulemt_cfg_current_key_str
                  {
                    \seq_set_eq:Nc \l_tmpa_seq
                      { c__kulemt_cfg_ \l_tmpa_str _values_seq }
                    \seq_if_exist:NT \l_tmpa_seq
                      {
                        \seq_if_in:NVF \l_tmpa_seq
                          \l__kulemt_cfg_current_value_str
                          {
                            \msg_error:nneee {kulemt} {cfg/unknown-value}
                              { \l__kulemt_cfg_current_value_str }
                              { \l__kulemt_cfg_current_key_str }
                              { ' \seq_use:Nnnn \l_tmpa_seq
                                { '~ and~ ' } { ',~ ' } { ',~ and~ ' } ' }
                            \tl_set:Ne \l__kulemt_cfg_current_value_str
                              { \seq_item:Nn \l_tmpa_seq {1} }
                          }
                      }
                    \bool_set_true:N \l_tmpa_bool
                  }
                  {
                    \str_if_eq:VnTF \l_tmpa_str {option}
                      {
                        \tl_if_empty:NF \l_tmpb_str
                          {
                            \seq_put_right:NV
                              \l__kulemt_cfg_current_options_seq \l_tmpb_str
                          }
                        \bool_set_true:N \l_tmpa_bool
                      }
                      { \bool_set_false:N \l_tmpa_bool }
                  }
                \bool_if:NTF \l_tmpa_bool
                  {
                    \prop_get:NVNF
                      \l__kulemt_cfg_current_section_prop
                      \l_tmpa_str
                      \l_tmpa_prop
                      { \prop_clear:N \l_tmpa_prop }
                    \prop_put:NVV \l_tmpa_prop \l_tmpb_str
                      \l__kulemt_cfg_current_value_str
                  }
                  {
                    \msg_warning:nnee {kulemt} {cfg/invalid-key}
                      { \l__kulemt_cfg_current_key_str }
                      { \l__kulemt_cfg_current_section_name_str }
                    \prop_set_eq:NN \l_tmpa_prop \q_no_value
                  }
              }
            \quark_if_no_value:NF \l_tmpa_prop
              {
                \prop_put:NVV \l__kulemt_cfg_current_section_prop \l_tmpa_str
                  \l_tmpa_prop
              }
          }
      }
  }
\msg_new:nnnn {kulemt} {cfg/invalid-key}
  { Invalid~ key~ '#1'~ in~ section~ '#2'. }
  { The~ invalid~ key~ is~ ignored. }
\msg_new:nnnn {kulemt} {cfg/unknown-value}
  { '#1'~ is~ not~ a~ valid~ value~ for~ key~ '#2'. }
  { Valid~ choices~ for~ '#2'~ are:~ #3. }
\cs_new_protected:Nn \__kulemt_cfg_get_section_item_from:nnN
  {
    \seq_if_in:NeTF \l_kulemt_masters_seq { \tl_to_str:n {#1} }
      { \prop_get:cnN { l__kulemt_cfg_section_ #1 _prop } {#2} #3 }
      {
        \msg_error:nneee {kulemt} {cfg/unknown-from}
          { \l__kulemt_cfg_current_key_str }
          { \l__kulemt_cfg_current_section_name_str } { #1 }
        \prop_set_eq:NN #3 \q_no_value
      }
  }
\cs_generate_variant:Nn \__kulemt_cfg_get_section_item_from:nnN { VV }
\msg_new:nnn {kulemt} {cfg/unknown-from}
  {
    The~ configuration~ key~ '#1'~ in~ section~ '#2'~ refers~ to~ '#3'.~
    But~ this~ section~ has~ not~ been~ defined~ before.
  }
\cs_new_protected:Nn \kulemt_set_master:n
  {
    \seq_if_in:NeF \l_kulemt_masters_seq { \tl_to_str:n {#1} }
      { \kulemt_read_config_file:n {#1} }
    \bool_lazy_or:nnT
      {
        ! \tl_if_eq_p:Nc \l_kulemt_master_prop
          { l__kulemt_cfg_master_ #1 _prop }
        }
      { \prop_if_empty_p:N \l_kulemt_master_prop }
      {
        \prop_set_eq:Nc \l_kulemt_master_prop { l__kulemt_cfg_master_ #1 _prop }
        \prop_if_empty:NT \l_kulemt_master_prop
          {
            \prop_set_eq:NN \l_tmpa_prop \l__kulemt_cfg_section_defaults_prop
            \prop_map_tokens:cn { l__kulemt_cfg_section_ #1 _prop }
              { \prop_put:Nnn \l_tmpa_prop }
            \prop_map_inline:Nn \l_tmpa_prop
              {
                \tl_set:Nn \l_tmpb_prop {##2}
                \prop_map_inline:Nn \l_tmpb_prop
                  {
                    \prop_put:Nen \l_kulemt_master_prop
                      { ##1 \tl_if_empty:nF {####1} { . ####1  } } { ####2 }
                  }
              }
            \prop_put:Nnn \l_kulemt_master_prop {abbreviation} {#1}
            \prop_set_eq:cN { l__kulemt_cfg_master_ #1 _prop }
              \l_kulemt_master_prop
          }
      }
  }
\cs_generate_variant:Nn \kulemt_set_master:n { V }
\cs_new_protected:Nn \kulemt_master_get_item:nN
  {
    \prop_if_empty:NT \l_kulemt_master_prop
      { \msg_fatal:nnn {kulemt} {cfg/master-not-set} {#1} }
    \exp_args:NNe \prop_get:NnNF \l_kulemt_master_prop {#1 . \languagename} #2
      {
        \str_if_eq:VnTF \languagename {british}
          { \prop_get:NnNF \l_kulemt_master_prop {#1 . english} #2 }
          { \use:n }
            { \prop_get:NnN \l_kulemt_master_prop {#1} #2 }
      }
    \quark_if_no_value:NF #2 { \__kulemt_cfg_str_with_utf:nN {#1} #2 }
  }
\msg_new:nnn {kulemt} {cfg/master-not-set}
  { A~ master~ must~ be~ selected~ before~ getting~ information~ about~ '#1'. }
\cs_new_protected:Nn \kulemt_master_get_item_or_fallback:nnN
  {
    \kulemt_master_get_item:nN {#1} #3
    \quark_if_no_value:NT #3
      {
        \tl_if_empty:nTF {#2}
          { \__kulemt_cfg_master_get_item_fallback:nN {#1} #3 }
          { \tl_set:Nn #3 {#2} }
      }
  }
\cs_new_protected:Nn \kulemt_master_get_required_item:nN
  {
    \kulemt_master_get_item:nN {#1} #2
    \quark_if_no_value:NT #2
      {
        \__kulemt_cfg_master_get_item_fallback:nN {#1} \l_tmpa_tl
        \prop_get:NnN \l_kulemt_master_prop {abbreviation} \l_tmpb_tl
        \msg_error:nneee {kulemt} {cfg/missing-key}
          { \l_tmpb_tl } {#1} { \l_tmpa_tl }
        \prop_put:NnV \l_kulemt_master_prop {#1} \l_tmpa_tl
        \prop_set_eq:cN { l__kulemt_cfg_master_ \l_tmpb_tl _prop }
          \l_kulemt_master_prop
      }
  }
\msg_new:nnnn {kulemt} {cfg/missing-key}
  { The~ required~ key~ '#2'~ is~ missing~ for~ master~ '#1'. }
  { If~ you~ continue,~ the~ value~ '#3'~ will~ be~ used. }
\cs_new_protected:Nn \__kulemt_cfg_master_get_item_fallback:nN
  {
    \seq_set_eq:Nc \l_tmpa_seq { c__kulemt_cfg_ #1 _values_seq }
    \tl_set:Ne #2 { \seq_if_exist:NTF \l_tmpa_seq
                      { \seq_item:Nn \l_tmpa_seq {1} }
                      { ???? } }
  }
\cs_new_protected:Nn \kulemt_master_get_faculty_name:N
  {
    \kulemt_master_get_item:nN {faculty} #1
    \bool_lazy_or:nnT
      { \quark_if_no_value_p:N #1 }
      { \str_if_eq_p:Vn #1 {multi} }
      { \tl_clear:N #1 }
    }
\prg_new_protected_conditional:Nnn \kulemt_master_obsolete_item:n { T, F, TF }
  { \tl_if_in:nnTF {#1} {.} { \prg_return_true: } { \prg_return_false: } }
\cs_if_exist:NTF \text_titlecase_first:nn
  {
    \cs_new_protected:Nn \kulemt_titlecase_first:n
      { \exp_args:NV \text_titlecase_first:nn \languagename {#1} }
  }
  {
    \cs_new_protected:Nn \kulemt_titlecase_first:n
      { \MakeUppercase #1 }
  }
\cs_generate_variant:Nn \kulemt_titlecase_first:n { V }
\cs_new_protected:Nn \kulemt_cfg_print_text:n
  {
    \__kulemt_cfg_get_text:n {#1}
    \l__kulemt_cfg_tmp_tl
  }
\cs_new_protected:Nn \kulemt_cfg_print_text_ucfirst:n
  {
    \__kulemt_cfg_get_text:n {#1}
    \kulemt_titlecase_first:V \l__kulemt_cfg_tmp_tl
  }
\cs_new_protected:Nn \__kulemt_cfg_get_text:n
  {
     \exp_args:NNe
      \prop_get:NnNF \l_kulemt_cfg_prop { text. #1 .\languagename }
        \l__kulemt_cfg_tmp_tl
        {
          \str_if_eq:VnT \languagename {british}
            {
              \prop_get:NnN \l_kulemt_cfg_prop { text. #1 .english }
                \l__kulemt_cfg_tmp_tl
            }
        }
    \quark_if_no_value:NT \l__kulemt_cfg_tmp_tl
      {
        \msg_error:nnee {kulemt} {cfg/text-not-set} {#1} { \languagename }
        \prop_get:NnN \l_kulemt_cfg_prop { text. #1 .english }
          \l__kulemt_cfg_tmp_tl
      }
    \quark_if_no_value:NTF \l__kulemt_cfg_tmp_tl
      { \tl_set:Ne \l__kulemt_cfg_tmp_tl {???} }
      { \__kulemt_cfg_str_with_utf:N \l__kulemt_cfg_tmp_tl }
  }
\msg_new:nnnn {kulemt} {cfg/text-not-set}
  { Configuration~ data~ 'text.#1'~ is~ undefined~ for~ language~ '#2'. }
  { Correct~ the~ configuration~ file.\\
    I~ shall~ use~ the~ English~ text~ for~ now~ if~ available. }
\cs_new_protected:Nn \kulemt_cfg_print_text_from_opt:n
  {
    \__kulemt_cfg_get_text_from_opt:n {#1}
    \l__kulemt_cfg_tmp_tl
  }
\cs_new_protected:Nn \kulemt_cfg_print_text_from_opt_ucfirst:n
  {
    \__kulemt_cfg_get_text_from_opt:n {#1}
    \kulemt_titlecase_first:V \l__kulemt_cfg_tmp_tl
  }
\cs_new_protected:Nn \__kulemt_cfg_get_text_from_opt:n
  {
    \exp_args:Ne \__kulemt_cfg_get_text:n
      {
        #1
        \int_compare:nNnF { \seq_count:c { l_kulemt_opt_ #1 _seq } } = {1}
          { .plural }
      }
  }
\cs_new_protected:Nn \kulemt_master_print_required_item:n
  {
    \kulemt_master_get_required_item:nN {#1} \l__kulemt_cfg_tmp_tl
    \l__kulemt_cfg_tmp_tl
  }
\cs_generate_variant:Nn \msg_error:nnn { nne }
\cs_generate_variant:Nn \msg_fatal:nnn { nne }
\cs_generate_variant:Nn \msg_fatal:nnnn { nnee }
\str_if_eq:VnT \encodingdefault {OT1}
  {
    \cs_set:Npn \encodingdefault {T1}
    \fontencoding { \encodingdefault }
  }
\str_if_eq:VnT \rmdefault {cmr}
  {
    \cs_set:Npn \rmdefault {lmr}
    \cs_set:Npn \sfdefault {lmss}
    \cs_set:Npn \ttdefault {lmtt}
    \fontfamily { \rmdefault }
  }
\bool_new:N \l__kulemt_opt_allow_class_option_bool
\bool_set_true:N \l__kulemt_opt_allow_class_option_bool
\cs_new_protected:Nn \__kulemt_opt_class_option:n
  {
    \bool_if:NTF \l__kulemt_opt_allow_class_option_bool
      { #1 }
      { \msg_error:nne {kulemt} {opt/not-class} { \kulemt_keys_key: } }
  }
\msg_new:nnnn {kulemt} {opt/not-class}
  { Option~ '#1'~ can~ only~ be~ used~ as~ a~ class~ option. }
  { This~ setup~ option~ will~ be~ ignored. }
\seq_new:N \l_kulemt_memoir_options_seq
\cs_new_protected:Nn \__kulemt_opt_memoir_option:n
  {
    \__kulemt_opt_class_option:n
      { \seq_put_right:Nn \l_kulemt_memoir_options_seq {#1} }
  }
\cs_new_protected:Nn \__kulemt_opt_memoir_option:
  { \exp_args:Ne \__kulemt_opt_memoir_option:n { \kulemt_keys_key: } }
\bool_new:N \l_kulemt_opt_article_bool
\keys_define:nn {kulemt}
  {
    article .code:n =
      {
        \__kulemt_opt_class_option:n
          {
            \bool_set_true:N \l_kulemt_opt_article_bool
            \__kulemt_opt_memoir_option:
          }
      } ,
    article .value_forbidden:n = true
  }
\keys_define:nn {kulemt}
  {
    twocolumn .code:n =
      {
        \__kulemt_opt_class_option:n
          {
            \bool_if:NTF \l_kulemt_opt_article_bool
              { \__kulemt_opt_memoir_option: }
              { \msg_error:nne {kulemt} {opt/not-art} { \kulemt_keys_key: } }
          }
      } ,
    twocolumn .value_forbidden:n = true
  }
\msg_new:nnnn {kulemt} {opt/not-art}
  { Option~ '#1'~ can~ only~ be~ used~ after~ the~ class~ option~ 'article'. }
  { This~ setup~ option~ will~ be~ ignored. }
\tl_new:N \l_kulemt_opt_cfgfile_tl
\keys_define:nn {kulemt}
  {
    cfgfile .code:n =
      {
        \__kulemt_opt_class_option:n
          { \tl_set:Nn \l_kulemt_opt_cfgfile_tl {#1} }

      } ,
    cfgfile .value_required:n = true,
    cfgfile .initial:n = { kulemt.ini }
  }
\tl_new:N \l_kulemt_opt_master_tl
\keys_define:nn {kulemt}
  {
    master .code:n =
      {
        \__kulemt_opt_class_option:n
          { \tl_set:Nn \l_kulemt_opt_master_tl {#1} }
      } ,
    master .value_required:n = true
  }
\int_new:N \l_kulemt_opt_ptsize_int
\int_set:Nn \l_kulemt_opt_ptsize_int {11}
\keys_define:nn {kulemt}
  {
    10pt .code:n =
      {
        \__kulemt_opt_class_option:n
          { \int_set:Nn \l_kulemt_opt_ptsize_int {10} }
      } ,
    10pt .value_forbidden:n = true,
    11pt .code:n =
      {
        \__kulemt_opt_class_option:n
          { \int_set:Nn \l_kulemt_opt_ptsize_int {11} }
      } ,
    11pt .value_forbidden:n = true
  }
\__kulemt_opt_memoir_option:n {a4paper}
\keys_define:nn {kulemt} {
    draft .code:n = { \__kulemt_opt_memoir_option: } ,
    draft .value_forbidden:n = true ,
    openany .code:n = { \__kulemt_opt_memoir_option: } ,
    openany .value_forbidden:n = true,
    openleft .code:n = { \__kulemt_opt_memoir_option: } ,
    openleft .value_forbidden:n = true,
    openright .code:n = { \__kulemt_opt_memoir_option:n {open} } ,
    openright .value_forbidden:n = true
  }
\bool_new:N \l_kulemt_opt_lrequal_bool
\bool_set_true:N \l_kulemt_opt_lrequal_bool
\keys_define:nn {kulemt} {
    oneside .code:n =
      {
        \__kulemt_opt_class_option:n
          {
            \seq_put_right:Nn \l_kulemt_memoir_options_seq {oneside}
            \bool_set_true:N \l_kulemt_opt_lrequal_bool
          }
      } ,
    oneside .value_forbidden:n = true,
    twoside .code:n =
      {
        \__kulemt_opt_class_option:n
          {
            \seq_put_right:Nn \l_kulemt_memoir_options_seq {twoside}
            \bool_set_false:N \l_kulemt_opt_lrequal_bool
          }
      } ,
    twoside .value_forbidden:n = true,
    twosidelrequal .code:n =
      {
        \__kulemt_opt_class_option:n
          {
            \seq_put_right:Nn \l_kulemt_memoir_options_seq {twoside}
            \bool_set_true:N \l_kulemt_opt_lrequal_bool
          }
      } ,
    twosidelrequal .value_forbidden:n = true
  }
\dim_new:N \l_kulemt_opt_bind_dim
\keys_define:nn {kulemt}
  {
    bind .code:n =
      {
        \__kulemt_opt_class_option:n
          { \dim_set:Nn \l_kulemt_opt_bind_dim {#1} }
      } ,
    bind .value_required:n = true
  }
\seq_new:N \l_kulemt_babel_seq
\tl_new:N \l_kulemt_language_tl
\cs_new_protected:Nn \__kulemt_opt_store_language:n
  {
    \__kulemt_opt_class_option:n
      {
        \tl_if_empty:NTF \l_kulemt_language_tl
          { \tl_set:Nn \l_kulemt_language_tl {#1} }
          {
            \msg_fatal:nnee {kulemt} {opt/multiple-languages}
              { \l_kulemt_language_tl } {#1}
          }
      }
  }
\msg_new:nnnn {kulemt} {opt/multiple-languages}
  {
    You~ can~ set~ the~ main~ text~ language~ only~ once.\\
    You~ used~ the~ language~ options~ '#1'~ and~ '#2'.
  }
  { Remove~ one~ of~ the~ language~ options. }
\keys_define:nn {kulemt}
  {
    dutch .code:n =   { \__kulemt_opt_store_language:n {dutch} } ,
    dutch .value_forbidden:n = true,
    english .code:n = { \__kulemt_opt_store_language:n {english} } ,
    english .value_forbidden:n = true,
    british .code:n = { \__kulemt_opt_store_language:n {british} } ,
    british .value_forbidden:n = true
  }
\keys_define:nn {kulemt}
  {
    extralanguage .code:n =
      {
        \__kulemt_opt_class_option:n
          {
            \tl_if_empty:nF {#1}
              { \seq_put_right:Nn \l_kulemt_babel_seq {#1} }
          }
      } ,
    extralanguage .value_required:n = true
  }
\keys_define:nn {kulemt}
  {
    fleqn .code:n = { \__kulemt_opt_memoir_option: } ,
    fleqn .value_forbidden:n = true,
    oldfontcommands .code:n = { \__kulemt_opt_memoir_option: } ,
    oldfontcommands .value_forbidden:n = true
  }
\cs_new_protected:Nn \__kulemt_opt_check_required:nn
  {
    \exp_args:Ne \kulemt_at_end_preamble:n
      {
        \exp_not:N \bool_if:NF \exp_not:N \l_kulemt_opt_article_bool
          {
            \exp_not:c { #1 _if_empty:NT } \exp_not:c { l_kulemt_opt_ #2 _ #1 }
              { \exp_not:N \msg_fatal:nnn {kulemt} {opt/missing} {#2} }
          }
      }
  }
\msg_new:nnn {kulemt} {opt/missing} { A~ required~ option~ '#1'~ is~ missing. }
\seq_new:N \l__kulemt_opt_masteroption_seq
\keys_define:nn {kulemt}
  {
    masteroption .code:n =
      {
        \tl_if_empty:nF {#1}
          {
            \seq_if_in:NnF \l__kulemt_opt_masteroption_seq {#1}
              { \seq_put_right:Nn \l__kulemt_opt_masteroption_seq {#1} }
          }
      } ,
    masteroption .value_required:n = true
  }
\seq_new:N \l_kulemt_opt_masteroption_seq
\cs_new_protected:Nn \__kulemt_opt_check_masteroption:
  {
    \tl_if_empty:NTF \l_kulemt_opt_master_tl
      {
        \seq_set_eq:NN \l_kulemt_opt_masteroption_seq
          \l__kulemt_opt_masteroption_seq
      }
      {
        \seq_clear:N \l_kulemt_opt_masteroption_seq
        \kulemt_master_get_item_or_fallback:nnN {option} {} \l_tmpa_tl
        \str_if_eq:VnTF \l_tmpa_tl {forbidden}
          {
            \seq_if_empty:NF \l__kulemt_opt_masteroption_seq
              {
                \msg_warning:nn {kulemt} {opt/masteroption-forbidden}
                \seq_clear:N \l__kulemt_opt_masteroption_seq
              }
          }
          {
            \str_if_eq:VnT \l_tmpa_tl {required}
              {
                \seq_if_empty:NT \l__kulemt_opt_masteroption_seq
                  {
                    \msg_fatal:nne {kulemt} {opt/masteroption-missing}
                      { \l_kulemt_opt_master_tl }
                  }
              }
            \seq_map_inline:Nn \l__kulemt_opt_masteroption_seq
              {
                \kulemt_master_get_item:nN { option. ##1  } \l_tmpa_tl
                \quark_if_no_value:NT \l_tmpa_tl
                  {
                    \msg_info:nnn {kulemt} {opt/masteroption-no-abbrev} {##1}
                    \tl_set:Nn \l_tmpa_tl {##1}
                  }
                \seq_put_right:NV \l_kulemt_opt_masteroption_seq \l_tmpa_tl
              }
          }
      }
  }
\kulemt_at_end_preamble:n { \__kulemt_opt_check_masteroption: }
\msg_new:nnn {kulemt} {opt/masteroption-forbidden}
  {
    The~ option~ 'masteroption'~ is~ ignored~ because~ your~ program~
    disallows~ a~ master's~ program~ option~ on~ front~ pages.
  }
\msg_new:nnn {kulemt} {opt/masteroption-missing}
  {
    For~ master~ '#1'~ you~ must~ specify~ at~ least~ one~
    master's~ program~ option.
  }
\msg_new:nnn {kulemt} {opt/masteroption-no-abbrev}
  {
    The~ master~ option~ '#1'~ is~ not~ a~ known~ abbreviation.\\
    It~ is~ used~ directly~ as~ the~ master~ option~ text.
   }
\cs_new_protected:Nn \__kulemt_opt_seq_add_split:NnnN
  {
    \seq_set_split:Nnn \l_tmpa_seq {#3} {#2}
    \seq_remove_all:Nn \l_tmpa_seq {}
    \seq_if_empty:NTF \l_tmpa_seq
      {
        \bool_if:NT #4
          {
            \seq_clear:N #1
            \seq_put_right:Nn #1 {}
          }
      }
      {
        \seq_if_empty:NTF #1
          { \seq_set_eq:NN #1 \l_tmpa_seq }
          {
            \seq_get:NN #1 \l_tmpa_tl
            \tl_if_empty:NTF \l_tmpa_tl
              { \seq_set_eq:NN #1 \l_tmpa_seq }
              { \seq_concat:NNN #1 #1 \l_tmpa_seq }
          }
      }
  }
\bool_const:Nn \c__kulemt_opt_disallow_empty_bool { \c_false_bool }
\bool_const:Nn \c__kulemt_opt_allow_empty_bool    { \c_true_bool  }
\tl_new:N \l_kulemt_opt_title_tl
\keys_define:nn {kulemt}
  {
    title .code:n =
      {
        \tl_set:Nn \l_kulemt_opt_title_tl {#1}
        \kulemt_at_end_preamble:n { \title{#1} }
      } ,
    title .value_required:n = true
  }
\__kulemt_opt_check_required:nn {tl} {title}
\tl_new:N \l_kulemt_opt_subtitle_tl
\keys_define:nn {kulemt}
  {
    subtitle .tl_set:N = \l_kulemt_opt_subtitle_tl ,
    subtitle .value_required:n = true
  }
\seq_new:N \l_kulemt_opt_author_seq
\keys_define:nn {kulemt}
  {
    author .code:n =
      {
        \__kulemt_opt_seq_add_split:NnnN \l_kulemt_opt_author_seq
          {#1} { \and } \c__kulemt_opt_disallow_empty_bool
      } ,
    author .value_required:n = true
  }
\__kulemt_opt_check_required:nn {seq} {author}
\kulemt_at_end_preamble:n
  {
    \seq_if_empty:NF \l_kulemt_opt_author_seq
      {
        \exp_args:Ne \author
          { \seq_use:Nn \l_kulemt_opt_author_seq { \and } }
      }
  }
\seq_new:N \l_kulemt_opt_promoter_seq
\keys_define:nn {kulemt}
  {
    promoter .code:n =
      {
        \__kulemt_opt_seq_add_split:NnnN \l_kulemt_opt_promoter_seq
          {#1} { \and } \c__kulemt_opt_disallow_empty_bool
      } ,
    promoter .value_required:n = true ,
    promotor .meta:n = { promoter = {#1} } ,
    promotor .value_required:n = true
  }
\__kulemt_opt_check_required:nn {seq} {promoter}
\seq_new:N \l_kulemt_opt_assessor_seq
\keys_define:nn {kulemt}
  {
    assessor .code:n =
      {
        \__kulemt_opt_seq_add_split:NnnN \l_kulemt_opt_assessor_seq
          {#1} { \and } \c__kulemt_opt_allow_empty_bool
      } ,
      assessor .value_required:n = true
  }
\__kulemt_opt_check_required:nn {seq} {assessor}
\seq_new:N \l_kulemt_opt_assistant_seq
\keys_define:nn {kulemt}
  {
    assistant .code:n =
      {
        \__kulemt_opt_seq_add_split:NnnN \l_kulemt_opt_assistant_seq
          {#1} { \and } \c__kulemt_opt_allow_empty_bool
      } ,
    assistant .value_required:n = true
  }
\__kulemt_opt_check_required:nn {seq} {assistant}
\int_new:N \l_kulemt_opt_acyear_int
\int_set:Nn \l_kulemt_opt_acyear_int
  { \c_sys_year_int \int_compare:nNnT { \c_sys_month_int } < {10} { - 1 } }
\keys_define:nn {kulemt}
  {
    acyear .code:n =
      {
        \tl_if_empty:nF {#1}
          {
            \regex_extract_once:nnN { \A \d{4} } {#1} \l_tmpa_seq
            \seq_if_empty:NTF \l_tmpa_seq
              { \msg_error:nnn {kulemt} {opt/invalid-year} {#1} }
              {
                \int_set:Nn \l_kulemt_opt_acyear_int
                  { \seq_item:Nn \l_tmpa_seq {1} }
              }
          }
      } ,
    acyear .value_required:n = true
  }
\msg_new:nnn {kulemt} {opt/invalid-year}
  {
    Value~ '\tl_to_str:n{#1}'~ of~ option~ 'acyear'~ does~ not~ start~ with~
    a~ 4-digit~ number.
  }
\bool_new:N \l_kulemt_include_coverpage_bool
\bool_new:N \l_kulemt_include_frontpages_bool
\bool_new:N \l_kulemt_include_text_bool
\bool_set_true:N \l_kulemt_include_frontpages_bool
\bool_set_true:N \l_kulemt_include_text_bool
\keys_define:nn {kulemt}
  {
    coverpageonly .code:n =
      {
        \bool_set_true:N  \l_kulemt_include_coverpage_bool
        \bool_set_false:N \l_kulemt_include_frontpages_bool
        \bool_set_false:N \l_kulemt_include_text_bool
      } ,
    coverpageonly .value_forbidden:n = true ,
    frontpagesonly .code:n =
      {
        \bool_set_false:N \l_kulemt_include_coverpage_bool
        \bool_set_true:N  \l_kulemt_include_frontpages_bool
        \bool_set_false:N \l_kulemt_include_text_bool
      } ,
    frontpagesonly .value_forbidden:n = true
  }
\NewDocumentCommand \setup { m }
  {
    \bool_set_false:N \l__kulemt_opt_allow_class_option_bool
    \keys_set:nn {kulemt} {#1}
  }
\@onlypreamble \setup
\tl_new:N \l_kulemt_master_language_tl
\cs_new_protected:Nn \kulemt_process_class_options:
  {
    \bool_set_true:N \l__kulemt_opt_allow_class_option_bool
    \ProcessKeyOptions \scan_stop:
    \tl_if_empty:NTF \l_kulemt_opt_master_tl
      {
        \bool_if:NF \l_kulemt_opt_article_bool
          { \msg_fatal:nn {kulemt} {opt/master-missing} }
        \tl_if_empty:NT \l_kulemt_language_tl
          { \tl_set:Nn \l_kulemt_language_tl {english} }
        \seq_put_right:NV \l_kulemt_babel_seq \l_kulemt_language_tl
      }
      {
        \kulemt_set_master:V \l_kulemt_opt_master_tl
        \kulemt_master_get_required_item:nN {language} \l_tmpa_str
        \tl_set_rescan:Nno \l_kulemt_master_language_tl {} { \l_tmpa_str }
        \tl_if_empty:NT \l_kulemt_language_tl
          { \tl_set_eq:NN \l_kulemt_language_tl \l_kulemt_master_language_tl }
        \bool_lazy_and:nnF
          { \str_if_eq_p:Vn \l_kulemt_language_tl {british} }
          { \str_if_eq_p:Vn \l_kulemt_master_language_tl {english} }
          { \seq_put_right:NV \l_kulemt_babel_seq \l_kulemt_master_language_tl }
        \str_if_eq:VVF \l_kulemt_language_tl \l_kulemt_master_language_tl
          { \seq_put_right:NV \l_kulemt_babel_seq \l_kulemt_language_tl }
      }
  }
\msg_new:nnn {kulemt} {opt/master-missing}
  { The~ required~ document~ class~ option~ 'master'~ is~ missing. }
\cs_generate_variant:Nn \msg_fatal:nnn { nne }
\cs_generate_variant:Nn \seq_put_left:Nn { Ne }
\kulemt_process_class_options:
\seq_put_left:Ne \l_kulemt_memoir_options_seq
  { \int_use:N \l_kulemt_opt_ptsize_int pt }
\exp_last_unbraced:Ne \LoadClass
  { [ \seq_use:Nn \l_kulemt_memoir_options_seq {,} ] } {memoir} [2018/04/04]
\exp_last_unbraced:Ne \RequirePackage
  { [ \seq_use:Nn \l_kulemt_babel_seq {,} , main = \l_kulemt_language_tl ] }
  {babel}
\cs_new_protected:Nn \kulemt_selectlanguage:n
  {
    \tl_if_empty:nF {#1}
      {
        \str_if_eq:nnTF {#1} {english}
          {
            \seq_if_in:NnTF \l_kulemt_babel_seq {british}
              { \selectlanguage {british} }
              { \selectlanguage {#1} }
          }
          { \selectlanguage {#1} }
      }
  }
\cs_generate_variant:Nn \kulemt_selectlanguage:n { o, V }
\cs_if_exist:NTF \captionsbritish
  { \tl_put_right:Nn \captionsbritish }
  {
    \cs_if_exist:NTF \captionsenglish
      { \tl_put_right:Nn \captionsenglish }
      { \use_none:n }
  }
    {
      \tl_set:Nn \appendixtocname  { Appendices }
      \tl_set:Nn \appendixpagename { Appendices }
      \tl_set:Nn \appendixrefname  { Appendix \nobreakspace }
      \tl_set:Nn \chapterrefname   { Chapter \nobreakspace }
      \tl_set:Nn \figurerefname    { Figure }
      \tl_set:Nn \pagerefname      { page }
      \tl_set:Nn \partrefname      { Part \nobreakspace }
      \tl_set:Nn \tablerefname     { Table }
    }
\cs_if_exist:NTF \captionsdutch
  { \tl_put_right:Nn \captionsdutch }
  { \use_none:n }
    {
      \tl_set:Nn \appendixtocname  { B\ij lagen }
      \tl_set:Nn \appendixpagename { B\ij lagen }
      \tl_set:Nn \appendixrefname  { b\ij lage \nobreakspace }
      \tl_set:Nn \chapterrefname   { hoofdstuk \nobreakspace }
      \tl_set:Nn \figurerefname    { figuur }
      \tl_set:Nn \pagerefname      { pagina }
      \tl_set:Nn \partrefname      { Deel \nobreakspace }
      \tl_set:Nn \tablerefname     { tabel }
    }
\RequirePackage {graphicx}
\tl_set:Nn \theHsubfigure { \theHfigure . \arabic {subfigure} }
\tl_set:Nn \theHsubtable  { \theHtable  . \arabic {subtable}  }
\clist_new:N  \l_kulemt_incompatible_clist
\clist_set:Nn \l_kulemt_incompatible_clist { polyglossia }
\clist_put_right:Nn \l_kulemt_incompatible_clist
  { a4, a4wide, anysize, fullpage, geometry, typearea }
\bool_if:NF \l_kulemt_opt_article_bool
  { \cs_gset_eq:NN \maketitle \scan_stop: }
\int_compare:nNnTF { \l_kulemt_opt_ptsize_int } = {10}
  {
    \dim_set:Nn \textwidth  { 13cm }
    \dim_set:Nn \textheight { 20cm }
  }
  {
    \dim_set:Nn \textwidth  { 14cm }
    \dim_set:Nn \textheight { 215mm }
  }
\dim_set:Nn \foremargin
  {
    ( \paperwidth - \textwidth - \l_kulemt_opt_bind_dim )
    * \bool_if:NTF \l_kulemt_opt_lrequal_bool {5} {6} /10
  }
\dim_set:Nn \spinemargin
  {
    ( \paperwidth - \textwidth - \l_kulemt_opt_bind_dim )
    * \bool_if:NTF \l_kulemt_opt_lrequal_bool {5} {4} /10
    + \l_kulemt_opt_bind_dim
  }
\dim_set:Nn \marginparwidth { 56pt }
\dim_set:Nn \marginparsep   { 1.2\onelineskip }
\dim_set_eq:NN \marginparpush \onelineskip
\setulmargins {*} {*} {1.2}
\setheaderspaces {*} { \headsep } {*}
\checkthelayout
\fixthelayout
\kulemt_at_end_preamble:n { \checkandfixthelayout }
\RenewDocumentCommand \clearforchapter {}
  {
    \legacy_if:nTF {@mainmatter}
      {
        \legacy_if:nTF {@openleft}
          { \cleartoverso }
          {
            \legacy_if:nTF {@openright}
              { \cleartorecto }
              { \clearpage    }
          }
      }
      { \clearpage }
  }
\makeoddhead {ruled} {} {}
  { \legacy_if:nTF {@mainmatter} { \rightmark } { \scshape \leftmark } }
\pagestyle {ruled}
\makepagestyle {nohead}
\makeevenfoot {nohead} { \thepage } {} {}
\makeoddfoot  {nohead} {} {} { \thepage }
\aliaspagestyle {chapter} {nohead}
\maxsecnumdepth {subsection}
\maxtocdepth {section}
\setrmarg { 2.55em plus 1fil }
\tl_set:Nn \tocheadstart { \clearforchapter \chapterheadstart }
\skip_set:Nn \cftbeforechapterskip { 1ex plus 1pt }
\captionnamefont { \scshape }
\captiontitlefont { \upshape }
\captionstyle [ \centering ] { \centerlastline }
\cs_new_protected:Nn \__kulemt_front_define_font:
  {
    \file_if_exist:nTF {tgheros.sty}
      {
        \sys_if_engine_opentype:TF
          {
            \exp_args:Nne \str_if_in:nnF {-TU-T1-OT1-} {-\encodingdefault-}
              { \msg_info:nnn {kulemt} {front/encoding} {TU} }
            \tl_set_eq:NN \l_tmpa_tl \encodingdefault
            \RequirePackage {fontspec}
            \newfontfamily \kulemt_front_font: {texgyreheros}
              [ Extension = .otf,
                UprightFont = *-regular, BoldFont = *-bold,
                ItalicFont = *-italic, BoldItalicFont = *-bolditalic ]
            \tl_set_eq:NN \encodingdefault \l_tmpa_tl
          }
          {
            \exp_args:Nne \str_if_in:nnF {-T1-OT1-} {-\encodingdefault-}
              { \msg_info:nnn {kulemt} {front/encoding} {T1} }
            \RequirePackage {textcomp}
            \DeclareFontFamily     {TS1} {kulemtfpf} {}
            \DeclareEncodingSubset {TS1} {kulemtfpf} {0}
            \DeclareFontShape      {TS1} {kulemtfpf} {m} {n} {<-> ts1-qhvr} {}
            \DeclareFontFamily {T1} {kulemtfpf} {}
            \DeclareFontShape  {T1} {kulemtfpf} {m}  {n}    {<-> ec-qhvr} {}
            \DeclareFontShape  {T1} {kulemtfpf} {m}  {it}   {<-> ec-qhvri} {}
            \DeclareFontShape  {T1} {kulemtfpf} {m}  {sc}   {<-> ec-qhvr-sc} {}
            \DeclareFontShape  {T1} {kulemtfpf} {m}  {scit} {<-> ec-qhvri-sc} {}
            \DeclareFontShape  {T1} {kulemtfpf} {b}  {it}   {<-> ec-qhvbi} {}
            \DeclareFontShape  {T1} {kulemtfpf} {b}  {n}    {<-> ec-qhvb} {}
            \DeclareFontShape  {T1} {kulemtfpf} {b}  {scit} {<-> ec-qhvbi-sc} {}
            \DeclareFontShape  {T1} {kulemtfpf} {b}  {sc}   {<-> ec-qhvb-sc} {}
            \DeclareFontShape  {T1} {kulemtfpf} {m}  {sl}
              {<-> ssub*kulemtfpf/m/it} {}
            \DeclareFontShape  {T1} {kulemtfpf} {m}  {scsl}
              {<-> ssub*kulemtfpf/m/scit} {}
            \DeclareFontShape  {T1} {kulemtfpf} {b}  {sl}
              {<-> ssub*kulemtfpf/b/it} {}
            \DeclareFontShape  {T1} {kulemtfpf} {b}  {scsl}
              {<-> ssub*kulemtfpf/b/scit} {}
            \DeclareFontShape  {T1} {kulemtfpf} {bx} {it}
              {<-> ssub*kulemtfpf/b/it} {}
            \DeclareFontShape  {T1} {kulemtfpf} {bx} {n}
              {<-> ssub*kulemtfpf/b/n} {}
            \DeclareFontShape  {T1} {kulemtfpf} {bx} {scit}
              {<-> ssub*kulemtfpf/b/scit} {}
            \DeclareFontShape  {T1} {kulemtfpf} {bx} {sc}
              {<-> ssub*kulemtfpf/b/sc} {}
            \cs_new:Nn \kulemt_front_font:
              { \fontencoding{T1} \fontfamily{kulemtfpf} \selectfont }
          }
      }
      {
        \bool_if:NTF \l_kulemt_opt_article_bool
          {
            \cs_new:Npn \kulemt_front_font:
              {
                \msg_error:nn {kulemt} {front/font-missing}
                \cs_gset_eq:NN \kulemt_front_font: \scan_stop:
              }
          }
          { \msg_fatal:nn {kulemt} {front/font-missing} }
      }
  }
\msg_new:nnn {kulemt} {front/font-missing}
  {
    The~ font~ 'TeX~ Gyre~ Heros'~ is~ not~ installed.\\
    It~ is~ needed~ for~ the~ front~ pages~ in~ thesis~ layout.
  }
\msg_new:nnn {kulemt} {front/encoding}
  {
    The~ front~ page~ cannot~ use~ encoding~ '\encodingdefault'.\\
    Font~ encoding~ '#1'~ is~ used~ instead.
  }
\kulemt_at_end_preamble:n { \__kulemt_front_define_font: }
\cs_new_protected:Nn \__kulemt_front_print_title_page:
  {
    \clearpage
    \setcounter {page} {-1}
    \thispagestyle {empty}
    \hbox:n {}
    \skip_vertical:n { 1cm - \uppermargin - \topskip - \baselineskip }
    \hbox_to_wd:nn { \hsize }
      {
        \skip_horizontal:n { 2cm - \spinemargin }
        \vbox_to_zero:n
          {
            \dim_set:Nn \hsize { 17cm }
            \skip_zero:N \lineskip
            \skip_zero:N \parskip
            \cs_if_exist:NT \microtypesetup
              { \microtypesetup { activate=false } }
            \fontsize {12} {14} \kulemt_front_font:
            \raggedright
            \kulemt_selectlanguage:V \l_kulemt_master_language_tl
            \noindent
            \skip_horizontal:n { -1cm }
            \vbox_to_ht:nn { 3cm }
              {
                \kulemt_master_get_item_or_fallback:nnN
                  {faculty.logo} {logokul} \l_tmpa_tl
                \exp_args:NV \includegraphics \l_tmpa_tl
                \vss
              }
            \skip_vertical:n { 40pt plus 2fill }
            \group_begin:
              \kulemt_selectlanguage:V \l_kulemt_language_tl
              \fontsize {24.88} {30} \selectfont
              \l_kulemt_opt_title_tl
              \par
              \tl_if_empty:NF \l_kulemt_opt_subtitle_tl
                {
                  \skip_vertical:n { 1em }
                  \fontsize {17.28} {22} \selectfont
                  \l_kulemt_opt_subtitle_tl
                  \par
                }
            \group_end:
            \skip_vertical:n { 40pt plus .3fill }
            \group_begin:
              \fontsize {14.4} {18} \selectfont
              \seq_use:Nn \l_kulemt_opt_author_seq { \\ }
              \par
            \group_end:
            \skip_vertical:n { 30pt plus 1fill }
            \noindent
            \hfill
            \vbox:n
              {
                \dim_set:Nn \hsize { .5\textwidth }
                \raggedleft
                \kulemt_cfg_print_text_ucfirst:n {title.pre} ~
                \kulemt_master_print_required_item:n {name}
                \seq_if_empty:NF \l_kulemt_opt_masteroption_seq
                  {
                    ,~
                    \seq_use:Nn \l_kulemt_opt_masteroption_seq
                      { ~ \kulemt_cfg_print_text:n {and} ~ }
                  }
                \par
                \__kulemt_front_print_people:n {promoter}
                \__kulemt_front_print_people:n {assessor}
                \__kulemt_front_print_people:n {assistant}
              }
            \skip_vertical:n { 20pt plus 2fill }
            \centering
            \fontsize {14.4} {18} \selectfont
            \kulemt_cfg_print_text_ucfirst:n {acyear.pre} ~
            \int_use:N \l_kulemt_opt_acyear_int
            \c_space_tl \textendash \c_space_tl
            \int_eval:n { \l_kulemt_opt_acyear_int + 1 }
            \par
            \skip_vertical:n { 15pt }
            \skip_vertical:n { -277mm }
          }
        \hss
      }
    \clearpage
  }
\cs_new_protected:Nn \__kulemt_front_print_people:n
  {
    \seq_if_empty:cF { l_kulemt_opt_ #1 _seq }
      {
        \medskip
        \group_begin:
          \fontsize {12} {14.5} \fontshape { \itdefault } \selectfont
          \kulemt_cfg_print_text_from_opt_ucfirst:n {#1} \par
          \skip_vertical:n { 2pt }
        \group_end:
        \seq_use:cn { l_kulemt_opt_ #1 _seq } { \\ }
        \par
      }
  }
\cs_new_protected:Nn \__kulemt_front_print_copyright_page:
  {
    \clearpage
    \thispagestyle {empty}
    \hbox:n {}
    \vfill
    \group_begin:
      \fontsize {10} {12} \kulemt_front_font:
      \raggedright
      \dim_zero:N \parindent
      \skip_set:Nn \parskip { .5\baselineskip }
      \kulemt_selectlanguage:V \l_kulemt_master_language_tl
      \textcopyright \c_space_tl
      \int_eval:n { \l_kulemt_opt_acyear_int + 1 }
      \c_space_tl KU \nobreakspace Leuven
      \kulemt_master_get_faculty_name:N \l_tmpa_tl
      \tl_if_empty:NF \l_tmpa_tl
        { \c_space_tl \textendash \c_space_tl \l_tmpa_tl }
      \\
      \kulemt_cfg_print_text_ucfirst:n {publisher.pre} \c_space_tl
      \seq_use:Nnnn \l_kulemt_opt_author_seq
        { ~ \kulemt_cfg_print_text:n {and} ~ }
        { , ~ }
        {
          \str_if_eq:VnF \languagename {dutch} { , }
          \c_space_tl \kulemt_cfg_print_text:n {and} ~
        }
      , \\
      \kulemt_master_print_required_item:n {contact.address}
      \par
      \kulemt_cfg_print_text:n {copyright} \par
      \bool_xor:nnT
        { \str_if_eq_p:Vn \l_kulemt_master_language_tl {dutch} }
        { \str_if_eq_p:Vn \l_kulemt_language_tl {dutch} }
        {
          \kulemt_selectlanguage:V \l_kulemt_language_tl
          \kulemt_cfg_print_text:n {copyright} \par
        }
    \group_end:
    \clearpage
  }
\kulemt_after_begin_document:n
  {
    \bool_if:NF \l_kulemt_opt_article_bool
      {
        \bool_if:NT \l_kulemt_include_coverpage_bool
          { \__kulemt_front_print_title_page: }
        \bool_if:NT \l_kulemt_include_frontpages_bool
          {
            \__kulemt_front_print_title_page:
            \__kulemt_front_print_copyright_page:
          }
        \bool_if:NF \l_kulemt_include_text_bool
          { \end {document} }
        \frontmatter
        \AtEndDocument { \__kulemt_front_forgot_mainmatter: }
      }
  }
\cs_new_protected:Nn \__kulemt_front_forgot_mainmatter:
  { \msg_error:nnn {kulemt} {mainmatter} }
\msg_new:nnn {kulemt} {mainmatter}
  { You~ forgot~ to~ use~ ' \token_to_str:N \mainmatter '. }
\tl_put_left:Nn \mainmatter
  { \cs_gset_eq:NN  \__kulemt_front_forgot_mainmatter: \prg_do_nothing: }
\cs_generate_variant:Nn \tl_put_right:Nn { Ne }
\NewDocumentEnvironment {preface} { o }
  {
    \chapter { \prefacename }
    \IfNoValueTF {#1}
      {
        \seq_if_empty:NTF \l_kulemt_opt_author_seq
          { \tl_clear:N \l__kulemt_extra_preface_author_tl }
          {
            \tl_set:Nn \l__kulemt_extra_preface_author_tl
              { \seq_use:Nn \l_kulemt_opt_author_seq { \\ } }
          }
      }
      { \tl_set:Nn \l__kulemt_extra_preface_author_tl {#1} }
  }
  {
    \par
    \tl_if_blank:VF \l__kulemt_extra_preface_author_tl
      {
        \bigskip
        \raggedleft
        \itshape
        \l__kulemt_extra_preface_author_tl
      }
    \vfill
    \clearpage
  }
\tl_new:N \l__kulemt_extra_preface_author_tl
\AtEndOfClass
  {
    \bool_if:NF \l_kulemt_opt_article_bool
      {
        \RenewDocumentEnvironment {abstract} {}
          { \chapter { \abstractname } }
          { \clearpage }
      }
  }
\NewDocumentEnvironment {abstract*} { O { \l_kulemt_master_language_tl } }
  {
    \kulemt_selectlanguage:o {#1}
    \abstract
  }
  { \endabstract }
\NewDocumentCommand \listoffiguresandtables { O{\listfiguresandtablesname} }
  {
    \chapter {#1}
    \tl_set:Nn \@lofmaketitle { \section* { \listfigurename } }
    \listoffigures*
    \tl_set:Nn \@lotmaketitle { \section* { \listtablename } }
    \listoftables*
  }
\tl_new:N \listfiguresandtablesname
\tl_set:Nn \listfiguresandtablesname { List~ of~ Figures~ and~ Tables }
\cs_if_exist:NTF \captionsbritish
  { \tl_put_right:Ne \captionsbritish }
  {
    \cs_if_exist:NTF \captionsenglish
      { \tl_put_right:Ne \captionsenglish }
      { \use_none:n }
  }
    {
      \exp_not:n { \tl_set:Nn \listfiguresandtablesname }
        { \listfiguresandtablesname }
    }
\cs_if_exist:NT \captionsdutch
  {
    \tl_put_right:Nn \captionsdutch
      {
        \tl_set:Nn \listfiguresandtablesname
          { L\ij st~ van~ figuren~ en~ tabellen }
      }
  }
\clist_map_inline:Nn \l_kulemt_incompatible_clist
  {
    \AtBeginPackage {#1}
      { \msg_fatal:nne {kulemt} {load/incompatible} {#1} }
      {}
  }
\msg_new:nnn {kulemt} {load/incompatible}
  { Package~ '#1'~ is~ incompatible~ with~ kulemt. }
\endinput
%%
%% End of file `kulemt.cls'.
